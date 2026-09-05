"""Async E2B desktop client built on the base ``e2b`` SDK.

The official ``e2b-desktop`` package is a thin synchronous wrapper around the
``desktop`` template: xdotool for input, scrot for screenshots, and
x11vnc + noVNC for the interactive stream. We replicate that surface on
``AsyncSandbox`` directly because the backend is fully async and the sync
package pins a conflicting pillow version.
"""

import asyncio
import base64
import contextlib
import secrets
import shlex
import string
from typing import Literal, Mapping, Optional

from e2b import AsyncSandbox, AsyncVolume, SandboxLifecycle
from pydantic import BaseModel

DESKTOP_TEMPLATE = "desktop"
HOME_PATH = "/home/user"
WORKSPACE_PATH = "/home/user/workspace"
# Where an expert's own computer sees its owning user's shared workspace. The
# expert's durable home is WORKSPACE_PATH, same as everyone else's.
SHARED_PATH = "/home/user/shared"
# Standard XFCE user dirs redirected into the volume so a person's natural
# desktop activity (browser downloads, files saved to the desktop) persists.
PERSISTENT_HOME_DIRS = ("Downloads", "Desktop", "Documents")
DISPLAY = ":0"
VNC_PORT = 5900
STREAM_PORT = 6080
SCREENSHOT_PATH = "/tmp/agpt_screenshot.png"
# The stream password lives next to x11vnc's hashed one so a resume can hand
# back the URL the user already holds instead of restarting the proxy.
STREAM_PASSWORD_PATH = f"{HOME_PATH}/.vnc/agpt_stream_password"
# Bound on the E2B volumes API (private beta) so a slow create cannot stall
# sandbox creation; the by-name mount fallback is the normal path anyway.
VOLUME_API_TIMEOUT_SECONDS = 10
_KILL_TIMEOUT_SECONDS = 10

_TYPE_CHUNK_SIZE = 25
_TYPE_DELAY_MS = 75
_READY_POLL_SECONDS = 0.5
# XFCE routinely takes 10-15 s to bring up xfwm4 on a cold start.
_READY_POLL_ATTEMPTS = 40

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
    # Every mount path that was attached (WORKSPACE_PATH plus, for an expert's
    # computer, SHARED_PATH). ``volume_mounted``/``volume_name`` describe the
    # WORKSPACE_PATH mount specifically.
    mounted_paths: list[str] = []


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
        volume_mounts: Optional[Mapping[str, str]] = None,
        template: str = DESKTOP_TEMPLATE,
        metadata: Optional[Mapping[str, str]] = None,
    ) -> tuple["DesktopSession", PersistenceInfo]:
        """Create a desktop sandbox.

        *volume_mounts* maps mount paths to durable volume names (see
        ``workspace_volume_mounts``); *metadata* is stamped on the sandbox so
        its owner can find it again through the E2B API if the cached id is
        lost.
        """
        sandbox, persistence = await _create_sandbox_with_volumes(
            volume_mounts, api_key, timeout_seconds, template, metadata
        )
        session = cls(sandbox)
        try:
            await session.ensure_display(width, height)
            # WORKSPACE_PATH always exists (blocks default their cwd to it),
            # mounted or not; mounted paths get their mkdir as well.
            paths = dict.fromkeys([WORKSPACE_PATH, *persistence.mounted_paths])
            await session.run_command(
                "mkdir -p " + " ".join(shlex.quote(p) for p in paths)
            )
            if persistence.volume_mounted:
                await session.ensure_persistent_home()
        except BaseException:
            # The box is on the meter but no caller has its id yet: kill it
            # rather than leak a sandbox that would bill until timeout and
            # then sit paused forever.
            with contextlib.suppress(Exception):
                await asyncio.wait_for(sandbox.kill(), timeout=_KILL_TIMEOUT_SECONDS)
            raise
        return session, persistence

    @classmethod
    async def connect(cls, sandbox_id: str, api_key: str) -> "DesktopSession":
        sandbox = await AsyncSandbox.connect(sandbox_id, api_key=api_key)
        return cls(sandbox)

    async def start_stream(self) -> DesktopStream:
        """Return the live stream URL, starting the VNC stack only if needed.

        A resume (or a second ``start_desktop`` call) must not restart x11vnc
        and noVNC: that would sever the stream the user is watching and rotate
        the password baked into the URL they already hold. When the proxy is
        still serving — E2B's pause/resume restores processes — the saved
        password is reused and the same URL comes back.
        """
        password = await self._running_stream_password()
        if password is None:
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
            await self.run_command(
                f"umask 077 && printf %s {shlex.quote(password)} "
                f"> {shlex.quote(STREAM_PASSWORD_PATH)}"
            )
        host = self.sandbox.get_host(STREAM_PORT)
        url = (
            f"https://{host}/vnc.html"
            f"?autoconnect=true&resize=scale&password={password}"
        )
        return DesktopStream(url=url, sandbox_id=self.sandbox_id)

    async def _running_stream_password(self) -> Optional[str]:
        """The saved stream password, only while noVNC is actually listening."""
        if not await self._check(f'netstat -tuln | grep -q ":{STREAM_PORT} "'):
            return None
        try:
            saved = await self.sandbox.files.read(STREAM_PASSWORD_PATH)
        except Exception:
            return None
        saved = saved.strip() if isinstance(saved, str) else ""
        return saved or None

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
        return await self.is_mounted(WORKSPACE_PATH)

    async def is_mounted(self, path: str) -> bool:
        return await self._check(f"mountpoint -q {shlex.quote(path)}")

    async def ensure_persistent_home(self) -> None:
        """Redirect the standard user dirs into the mounted volume.

        Downloads/Desktop/Documents become symlinks into ``WORKSPACE_PATH`` so
        files a person creates through the live desktop persist on the volume
        (surviving sandbox destroy) rather than living only in the ephemeral
        rootfs. Idempotent: safe to run on every sandbox that mounts the
        volume, migrating any pre-existing content in once.
        """
        dirs = " ".join(PERSISTENT_HOME_DIRS)
        script = (
            f"for d in {dirs}; do "
            f'mkdir -p {WORKSPACE_PATH}/"$d"; '
            f'if [ -d {HOME_PATH}/"$d" ] && [ ! -L {HOME_PATH}/"$d" ]; then '
            f'cp -an {HOME_PATH}/"$d"/. {WORKSPACE_PATH}/"$d"/ 2>/dev/null || true; '
            f'rm -rf {HOME_PATH}/"$d"; fi; '
            f'ln -sfn {WORKSPACE_PATH}/"$d" {HOME_PATH}/"$d"; '
            f"done"
        )
        await self.run_command(script)

    async def resources(self) -> Optional[tuple[int, float]]:
        """``(vCPU, RAM GiB)`` as E2B reports for this box, or ``None``.

        Templates are sized by E2B, not by us, so cost telemetry reads the
        real numbers instead of assuming them. Never raises: a failed lookup
        just means the meter falls back to its default sizing.
        """
        try:
            info = await asyncio.wait_for(
                self.sandbox.get_info(), timeout=_KILL_TIMEOUT_SECONDS
            )
            return info.cpu_count, info.memory_mb / 1024
        except Exception:
            return None

    async def pause(self) -> None:
        await self.sandbox.pause()

    async def kill(self) -> None:
        await self.sandbox.kill()

    async def _xdotool(self, args: str) -> None:
        await self.run_command(f"xdotool {args}")

    async def ensure_display(self, width: int, height: int) -> None:
        if await self._check("pgrep -x xfwm4"):
            return
        if not await self._check(f"xdpyinfo -display {DISPLAY}"):
            await self.sandbox.commands.run(
                f"Xvfb {DISPLAY} -ac -screen 0 {width}x{height}x24 -retro -dpi 96 "
                "-nolisten tcp -nolisten unix",
                background=True,
            )
            await self._wait_for(f"xdpyinfo -display {DISPLAY}")
        await self.sandbox.commands.run(
            "startxfce4", background=True, envs={"DISPLAY": DISPLAY}
        )
        # Without gating on the window manager, the first xdotool call blocks
        # 10-15 s inside a half-started XFCE instead of failing fast here.
        await self._wait_for("pgrep -x xfwm4")

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


def _sandbox_create_kwargs(
    api_key: str,
    timeout_seconds: int,
    template: str,
    metadata: Optional[Mapping[str, str]] = None,
) -> dict:
    kwargs: dict = {
        "template": template,
        "api_key": api_key,
        "timeout": timeout_seconds,
        "lifecycle": SandboxLifecycle(on_timeout="pause", auto_resume=True),
    }
    if metadata:
        kwargs["metadata"] = dict(metadata)
    return kwargs


async def resolve_volume(volume_name: str, api_key: str) -> "AsyncVolume | str":
    """Create *volume_name* if it does not exist yet, else mount it by name.

    E2B has no get-or-create, so the create is expected to fail on every run
    after the first and the by-name fallback is the normal path. Bounded so a
    slow volumes API cannot stall sandbox creation. Shared with the CoPilot
    shell (``copilot/tools/e2b_sandbox``) so both surfaces resolve identically.
    """
    try:
        return await asyncio.wait_for(
            AsyncVolume.create(volume_name, api_key=api_key),
            timeout=VOLUME_API_TIMEOUT_SECONDS,
        )
    except Exception:
        return volume_name


async def _create_sandbox_with_volumes(
    volume_mounts: Optional[Mapping[str, str]],
    api_key: str,
    timeout_seconds: int,
    template: str = DESKTOP_TEMPLATE,
    metadata: Optional[Mapping[str, str]] = None,
) -> tuple[AsyncSandbox, PersistenceInfo]:
    kwargs = _sandbox_create_kwargs(api_key, timeout_seconds, template, metadata)
    if not volume_mounts:
        return await AsyncSandbox.create(**kwargs), PersistenceInfo()

    paths = list(volume_mounts)
    volumes = await asyncio.gather(
        *(resolve_volume(volume_mounts[path], api_key) for path in paths)
    )
    mounts = dict(zip(paths, volumes))
    try:
        sandbox = await AsyncSandbox.create(**kwargs, volume_mounts=mounts)
    except Exception as mount_error:
        sandbox = await AsyncSandbox.create(**kwargs)
        return sandbox, PersistenceInfo(
            warning=(
                "Persistent volume unavailable (E2B volumes are in private beta); "
                f"using suspend/resume persistence only: {mount_error}"
            )
        )
    return sandbox, PersistenceInfo(
        volume_mounted=WORKSPACE_PATH in volume_mounts,
        volume_name=volume_mounts.get(WORKSPACE_PATH),
        mounted_paths=list(volume_mounts),
    )
