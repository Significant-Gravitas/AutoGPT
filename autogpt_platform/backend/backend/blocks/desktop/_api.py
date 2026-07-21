"""Async E2B desktop client built on the base ``e2b`` SDK.

The official ``e2b-desktop`` package is a thin synchronous wrapper around the
``desktop`` template: xdotool for input, scrot for screenshots, and
x11vnc + noVNC for the interactive stream. We replicate that surface on
``AsyncSandbox`` directly because the backend is fully async and the sync
package pins a conflicting pillow version.
"""

import asyncio
import base64
import secrets
import shlex
import string
from typing import Literal, Optional

from e2b import AsyncSandbox, AsyncVolume, SandboxLifecycle
from pydantic import BaseModel

DESKTOP_TEMPLATE = "desktop"
WORKSPACE_PATH = "/home/user/workspace"
DISPLAY = ":0"
VNC_PORT = 5900
STREAM_PORT = 6080
SCREENSHOT_PATH = "/tmp/agpt_screenshot.png"

_TYPE_CHUNK_SIZE = 25
_TYPE_DELAY_MS = 75
_READY_POLL_SECONDS = 0.5
_READY_POLL_ATTEMPTS = 20

_KEY_ALIASES = {
    "enter": "Return",
    "return": "Return",
    "esc": "Escape",
    "escape": "Escape",
    "backspace": "BackSpace",
    "delete": "Delete",
    "tab": "Tab",
    "space": "space",
    "up": "Up",
    "down": "Down",
    "left": "Left",
    "right": "Right",
    "home": "Home",
    "end": "End",
    "pageup": "Page_Up",
    "pagedown": "Page_Down",
    "cmd": "super",
    "win": "super",
}


class DesktopStream(BaseModel):
    kind: Literal["desktop_stream"] = "desktop_stream"
    url: str
    provider: Literal["e2b"] = "e2b"
    sandbox_id: str
    requires_auth: bool = False


class PersistenceInfo(BaseModel):
    volume_mounted: bool = False
    volume_name: Optional[str] = None
    warning: Optional[str] = None


def map_key(key: str) -> str:
    return _KEY_ALIASES.get(key.strip().lower(), key.strip())


class DesktopSession:
    def __init__(self, sandbox: AsyncSandbox):
        self.sandbox = sandbox

    @property
    def sandbox_id(self) -> str:
        return self.sandbox.sandbox_id

    @classmethod
    async def create(
        cls,
        api_key: str,
        timeout_seconds: int,
        width: int,
        height: int,
        volume_name: Optional[str],
    ) -> tuple["DesktopSession", PersistenceInfo]:
        sandbox, persistence = await _create_sandbox_with_volume(
            volume_name, api_key, timeout_seconds
        )
        session = cls(sandbox)
        await session._ensure_display(width, height)
        if persistence.volume_mounted:
            await session.run_command(f"mkdir -p {shlex.quote(WORKSPACE_PATH)}")
        return session, persistence

    @classmethod
    async def connect(cls, sandbox_id: str, api_key: str) -> "DesktopSession":
        sandbox = await AsyncSandbox.connect(sandbox_id, api_key=api_key)
        return cls(sandbox)

    async def start_stream(self) -> DesktopStream:
        password = "".join(
            secrets.choice(string.ascii_letters + string.digits) for _ in range(16)
        )
        await self.run_command(
            "pkill -f '[n]ovnc_proxy' || true; pkill -x x11vnc || true"
        )
        await self.run_command(
            f"mkdir -p ~/.vnc && x11vnc -storepasswd {password} ~/.vnc/passwd"
        )
        await self.run_command(
            f"x11vnc -bg -display {DISPLAY} -forever -wait 50 -shared "
            f"-rfbport {VNC_PORT} -usepw 2>/tmp/x11vnc_stderr.log"
        )
        await self.sandbox.commands.run(
            f"cd /opt/noVNC/utils && ./novnc_proxy --vnc localhost:{VNC_PORT} "
            f"--listen {STREAM_PORT} --web /opt/noVNC > /tmp/novnc.log 2>&1",
            background=True,
        )
        await self._wait_for(f'netstat -tuln | grep ":{STREAM_PORT} "')
        host = self.sandbox.get_host(STREAM_PORT)
        url = (
            f"https://{host}/vnc.html"
            f"?autoconnect=true&resize=scale&password={password}"
        )
        return DesktopStream(url=url, sandbox_id=self.sandbox_id)

    async def screenshot_base64(self) -> str:
        await self.run_command(f"scrot --pointer {SCREENSHOT_PATH}")
        data = await self.sandbox.files.read(SCREENSHOT_PATH, format="bytes")
        await self.run_command(f"rm -f {SCREENSHOT_PATH}")
        return base64.b64encode(data).decode()

    async def move_mouse(self, x: int, y: int) -> None:
        await self._xdotool(f"mousemove --sync {x} {y}")

    async def click(
        self, button: int, x: Optional[int], y: Optional[int], double: bool = False
    ) -> None:
        if x is not None and y is not None:
            await self.move_mouse(x, y)
        repeat = "--repeat 2 --delay 100 " if double else ""
        await self._xdotool(f"click {repeat}{button}")

    async def drag(self, from_x: int, from_y: int, to_x: int, to_y: int) -> None:
        await self._xdotool(f"mousemove --sync {from_x} {from_y}")
        await self._xdotool("mousedown 1")
        await self._xdotool(f"mousemove --sync {to_x} {to_y}")
        await self._xdotool("mouseup 1")

    async def scroll(self, direction: Literal["up", "down"], amount: int) -> None:
        button = 4 if direction == "up" else 5
        await self._xdotool(f"click --repeat {amount} {button}")

    async def type_text(self, text: str) -> None:
        for i in range(0, len(text), _TYPE_CHUNK_SIZE):
            chunk = text[i : i + _TYPE_CHUNK_SIZE]
            await self._xdotool(
                f"type --delay {_TYPE_DELAY_MS} -- {shlex.quote(chunk)}"
            )

    async def press(self, keys: list[str]) -> None:
        combo = "+".join(map_key(k) for k in keys if k.strip())
        await self._xdotool(f"key {combo}")

    async def run_command(
        self, command: str, cwd: Optional[str] = None, timeout: int = 60
    ):
        return await self.sandbox.commands.run(
            command, cwd=cwd, timeout=timeout, envs={"DISPLAY": DISPLAY}
        )

    async def is_workspace_mounted(self) -> bool:
        return await self._check(f"mountpoint -q {shlex.quote(WORKSPACE_PATH)}")

    async def pause(self) -> None:
        await self.sandbox.pause()

    async def kill(self) -> None:
        await self.sandbox.kill()

    async def _xdotool(self, args: str) -> None:
        await self.run_command(f"xdotool {args}")

    async def _ensure_display(self, width: int, height: int) -> None:
        if await self._check(f"xdpyinfo -display {DISPLAY}"):
            return
        await self.sandbox.commands.run(
            f"Xvfb {DISPLAY} -ac -screen 0 {width}x{height}x24 -retro -dpi 96 "
            "-nolisten tcp -nolisten unix",
            background=True,
        )
        await self._wait_for(f"xdpyinfo -display {DISPLAY}")
        await self.sandbox.commands.run(
            "startxfce4", background=True, envs={"DISPLAY": DISPLAY}
        )

    async def _check(self, command: str) -> bool:
        try:
            await self.sandbox.commands.run(command)
            return True
        except Exception:
            return False

    async def _wait_for(self, command: str) -> None:
        for _ in range(_READY_POLL_ATTEMPTS):
            if await self._check(command):
                return
            await asyncio.sleep(_READY_POLL_SECONDS)
        raise TimeoutError(f"Timed out waiting for: {command.split()[0]}")


def _sandbox_create_kwargs(api_key: str, timeout_seconds: int) -> dict:
    return {
        "template": DESKTOP_TEMPLATE,
        "api_key": api_key,
        "timeout": timeout_seconds,
        "lifecycle": SandboxLifecycle(on_timeout="pause", auto_resume=True),
    }


async def _create_sandbox_with_volume(
    volume_name: Optional[str], api_key: str, timeout_seconds: int
) -> tuple[AsyncSandbox, PersistenceInfo]:
    kwargs = _sandbox_create_kwargs(api_key, timeout_seconds)
    if not volume_name:
        return await AsyncSandbox.create(**kwargs), PersistenceInfo()

    volume: AsyncVolume | str
    try:
        volume = await AsyncVolume.create(volume_name, api_key=api_key)
    except Exception:
        # Creation fails when the volume already exists; mount by name instead.
        volume = volume_name

    try:
        sandbox = await AsyncSandbox.create(
            **kwargs, volume_mounts={WORKSPACE_PATH: volume}
        )
        return sandbox, PersistenceInfo(volume_mounted=True, volume_name=volume_name)
    except Exception as mount_error:
        sandbox = await AsyncSandbox.create(**kwargs)
        return sandbox, PersistenceInfo(
            warning=(
                "Persistent volume unavailable (E2B volumes are in private beta); "
                f"using suspend/resume persistence only: {mount_error}"
            )
        )
