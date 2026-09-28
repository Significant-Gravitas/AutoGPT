"""Async E2B desktop client built on the base ``e2b`` SDK.

The official ``e2b-desktop`` package is a thin synchronous wrapper around the
``desktop`` template: xdotool for input, scrot for screenshots, and
x11vnc + noVNC for the interactive stream. We replicate that surface on
``AsyncSandbox`` directly because the backend is fully async and the sync
package pins a conflicting pillow version.
"""

import asyncio
import contextlib
import secrets
import shlex
import string
from typing import Literal, Mapping, Optional

from e2b import AsyncSandbox, AsyncVolume, SandboxLifecycle
from pydantic import BaseModel

from backend.util.e2b_network import (
    EgressOwner,
    connect_sandbox,
    create_sandbox,
    kill_sandbox,
)

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
# The VNC stack runs as root.  The model's shell runs as ``user``, and a
# password that shell can read is a password it can print into a transcript,
# where the chat's share link would then carry a working desktop URL.  x11vnc
# takes its password from a root-only file that it deletes the moment it has
# read it, so once the stream is up nothing on the box holds the credential.
# What a resume needs to hand back the same URL is kept by the caller, off the
# box (see ``backend.copilot.computer``).
VNC_USER = "root"
VNC_PASSWORD_PATH = "/root/.vnc/stream_password"
# Root's logs live in root's directory, not /tmp: there the box's user could
# plant a file of the same name, and with fs.protected_regular set the kernel
# refuses even root an O_CREAT open of another user's file in a sticky
# directory, which is what a box that ran its stream as the user has.
_VNC_DIR = VNC_PASSWORD_PATH.rsplit("/", 1)[0]
_X11VNC_LOG = f"{_VNC_DIR}/x11vnc.log"
_X11VNC_ERROR_LOG = f"{_VNC_DIR}/x11vnc_stderr.log"
_NOVNC_LOG = f"{_VNC_DIR}/novnc.log"
# The password file goes too: x11vnc deletes it only once it has read it, and
# one that failed before that would leave the credential resting on the box.
_STOP_STREAM = (
    "pkill -f '[n]ovnc_proxy' || true; pkill -x x11vnc || true; "
    f"rm -f {shlex.quote(VNC_PASSWORD_PATH)}"
)
# Bound on the E2B volumes API (private beta) so a slow create cannot stall
# sandbox creation; the by-name mount fallback is the normal path anyway.
VOLUME_API_TIMEOUT_SECONDS = 10
_KILL_TIMEOUT_SECONDS = 10

_TYPE_CHUNK_SIZE = 25
_TYPE_DELAY_MS = 75
_READY_POLL_SECONDS = 0.5
# XFCE routinely takes 10-15 s to bring up xfwm4 on a cold start.
_READY_POLL_ATTEMPTS = 40


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
        *,
        owner: EgressOwner,
    ) -> tuple["DesktopSession", PersistenceInfo]:
        """Create a desktop sandbox.

        *volume_mounts* maps mount paths to durable volume names (see
        ``workspace_volume_mounts``); *metadata* is stamped on the sandbox so
        its owner can find it again through the E2B API if the cached id is
        lost; *owner* is who the egress proxy sees the box as.
        """
        sandbox, persistence = await _create_sandbox_with_volumes(
            volume_mounts, api_key, timeout_seconds, template, metadata, owner=owner
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
                await asyncio.wait_for(
                    kill_sandbox(sandbox), timeout=_KILL_TIMEOUT_SECONDS
                )
            raise
        return session, persistence

    @classmethod
    async def connect(
        cls,
        sandbox_id: str,
        api_key: str,
        timeout_seconds: Optional[int] = None,
        *,
        owner: EgressOwner,
    ) -> "DesktopSession":
        """Reattach to a desktop; *timeout_seconds* re-arms its running-time
        limit, otherwise the SDK's 300 s default would pause a resumed desktop
        under the user long before a freshly created one."""
        sandbox = await connect_sandbox(
            AsyncSandbox, sandbox_id, owner, api_key=api_key, timeout=timeout_seconds
        )
        return cls(sandbox)

    async def start_stream(
        self, password: Optional[str] = None
    ) -> tuple[DesktopStream, str]:
        """Return the live stream URL and its password, starting the VNC stack
        only if needed.

        *password* is the one this caller issued last time.  While x11vnc and
        noVNC are both still serving it (E2B's pause/resume restores
        processes) the same URL comes back: restarting them would sever the
        stream the user is watching.  Without it, or once either is gone, the
        stack is (re)started under a fresh password, and whoever held the old
        URL is locked out.  The caller decides when to forget the password (after a
        pause, say) and so when a URL that may have leaked stops working.
        """
        if password is None or not await self._stream_listening():
            password = "".join(
                secrets.choice(string.ascii_letters + string.digits) for _ in range(16)
            )
            await self._vnc_command(_STOP_STREAM)
            await self._vnc_command(
                f"umask 077 && mkdir -p {shlex.quote(_VNC_DIR)}"
                f" && printf %s {shlex.quote(password)} > {shlex.quote(VNC_PASSWORD_PATH)}"
            )
            # -noshm: x11vnc runs as root and Xvfb as the box's user, and the X
            # server cannot attach a shared-memory segment that root owns
            # (MIT-SHM BadAccess, x11vnc exits 1).  Without it the screen is
            # read over the X socket instead.
            try:
                await self._vnc_command(
                    f"x11vnc -bg -noshm -display {DISPLAY} -forever -wait 50 "
                    f"-shared -rfbport {VNC_PORT} "
                    f"-passwdfile rm:{VNC_PASSWORD_PATH} "
                    f">{_X11VNC_LOG} 2>{_X11VNC_ERROR_LOG}"
                )
                # -bg returns once x11vnc has detached; one that dies after
                # that would leave noVNC serving a stream with nothing behind.
                await self._wait_for(f'netstat -tln | grep ":{VNC_PORT} "')
            except Exception as exc:
                # One that detached but never served is still running.
                with contextlib.suppress(Exception):
                    await self._vnc_command(_STOP_STREAM)
                # x11vnc's own words are in the box, not in the exception.
                raise RuntimeError(
                    f"x11vnc did not start: {await self._tail(_X11VNC_ERROR_LOG)}"
                ) from exc
            try:
                await self.sandbox.commands.run(
                    f"cd /opt/noVNC/utils && ./novnc_proxy --vnc localhost:{VNC_PORT} "
                    f"--listen {STREAM_PORT} --web /opt/noVNC > {_NOVNC_LOG} 2>&1",
                    background=True,
                    user=VNC_USER,
                )
                await self._wait_for(f'netstat -tuln | grep ":{STREAM_PORT} "')
            except Exception as exc:
                # Do not leave an x11vnc serving under a password nobody holds.
                with contextlib.suppress(Exception):
                    await self._vnc_command(_STOP_STREAM)
                raise RuntimeError(
                    f"noVNC did not start: {await self._tail(_NOVNC_LOG)}"
                ) from exc
        host = self.sandbox.get_host(STREAM_PORT)
        url = (
            f"https://{host}/vnc.html"
            f"?autoconnect=true&resize=scale&password={password}"
        )
        return DesktopStream(url=url, sandbox_id=self.sandbox_id), password

    async def _tail(self, path: str) -> str:
        try:
            result = await self._vnc_command(f"tail -n 15 {shlex.quote(path)}")
            return result.stdout.strip() or "(empty log)"
        except Exception:
            return "(log unreadable)"

    async def _stream_listening(self) -> bool:
        """Whether the whole stream is still up (it survives a pause).

        Both halves, not just the proxy: noVNC keeps listening after x11vnc
        has died, and a URL handed back then opens onto nothing.  A listener
        on the VNC port is not proof of x11vnc either: once it is gone the
        box's user can bind that port itself, so root's own process is asked
        for as well, which that user cannot forge.
        """
        try:
            await self._vnc_command(
                f"pgrep -x -u {VNC_USER} x11vnc >/dev/null"
                f' && netstat -tln | grep -q ":{VNC_PORT} "'
                f' && netstat -tln | grep -q ":{STREAM_PORT} "'
            )
        except Exception:
            return False
        return True

    async def _vnc_command(self, command: str):
        return await self.run_command(command, user=VNC_USER)

    async def run_command(
        self,
        command: str,
        cwd: Optional[str] = None,
        timeout: int = 60,
        user: Optional[str] = None,
    ):
        return await self.sandbox.commands.run(
            command, cwd=cwd, timeout=timeout, envs={"DISPLAY": DISPLAY}, user=user
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

    async def pause(self) -> None:
        await self.sandbox.pause()

    async def kill(self) -> None:
        await kill_sandbox(self.sandbox)

    async def stop_stream(self) -> None:
        """Stop serving the screen; the display itself stays up.

        Whatever password the stream ran under stops working with it, and the
        next ``start_stream`` brings the stack back under a fresh one.
        """
        await self._vnc_command(_STOP_STREAM)

    async def ensure_display(self, width: int, height: int) -> None:
        if await self._check("pgrep -x xfwm4"):
            return
        # Both daemons must write to files, never to the command's pipes: a
        # background command's stdout/stderr stream to the SDK handle, and
        # once that handle is dropped the pipe closes, so the first warning
        # Xvfb logs afterwards (Chrome opening a second window is enough)
        # kills it with SIGPIPE and every X client with it.
        if not await self._check(f"xdpyinfo -display {DISPLAY}"):
            await self.sandbox.commands.run(
                f"Xvfb {DISPLAY} -ac -screen 0 {width}x{height}x24 -retro -dpi 96 "
                "-nolisten tcp -nolisten unix > /tmp/xvfb.log 2>&1",
                background=True,
            )
            await self._wait_for(f"xdpyinfo -display {DISPLAY}")
        await self.sandbox.commands.run(
            "startxfce4 > /tmp/xfce.log 2>&1",
            background=True,
            envs={"DISPLAY": DISPLAY},
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


# Client-side deadline on each create call (E2B provisions in 5-15 s), and
# the number of mounted attempts before the volumes are given up on.  A
# desktop keeps its volumes for life once created, so one transient failure
# must not decide that for it.
CREATE_TIMEOUT_SECONDS = 30
MOUNTED_CREATE_ATTEMPTS = 2


async def _create_sandbox_with_volumes(
    volume_mounts: Optional[Mapping[str, str]],
    api_key: str,
    timeout_seconds: int,
    template: str = DESKTOP_TEMPLATE,
    metadata: Optional[Mapping[str, str]] = None,
    *,
    owner: EgressOwner,
) -> tuple[AsyncSandbox, PersistenceInfo]:
    kwargs = _sandbox_create_kwargs(api_key, timeout_seconds, template, metadata)
    if not volume_mounts:
        sandbox = await asyncio.wait_for(
            create_sandbox(AsyncSandbox, owner, **kwargs),
            timeout=CREATE_TIMEOUT_SECONDS,
        )
        return sandbox, PersistenceInfo()

    paths = list(volume_mounts)
    volumes = await asyncio.gather(
        *(resolve_volume(volume_mounts[path], api_key) for path in paths)
    )
    mounts = dict(zip(paths, volumes))
    mount_error: Exception | None = None
    for attempt in range(1, MOUNTED_CREATE_ATTEMPTS + 1):
        try:
            sandbox = await asyncio.wait_for(
                create_sandbox(AsyncSandbox, owner, **kwargs, volume_mounts=mounts),
                timeout=CREATE_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            mount_error = exc
            if attempt < MOUNTED_CREATE_ATTEMPTS:
                await asyncio.sleep(attempt)
            continue
        return sandbox, PersistenceInfo(
            volume_mounted=WORKSPACE_PATH in volume_mounts,
            volume_name=volume_mounts.get(WORKSPACE_PATH),
            mounted_paths=list(volume_mounts),
        )

    # The volumes really are unavailable: fall back to a volume-less box and
    # say so in its stamp, so the Computer tab does not report mounts it
    # does not have.
    if kwargs.get("metadata"):
        kwargs["metadata"] = {**kwargs["metadata"], "autogpt_mounts": "none"}
    sandbox = await asyncio.wait_for(
        create_sandbox(AsyncSandbox, owner, **kwargs), timeout=CREATE_TIMEOUT_SECONDS
    )
    return sandbox, PersistenceInfo(
        warning=(
            "Persistent volume unavailable (E2B volumes are in private beta); "
            f"using suspend/resume persistence only: {mount_error}"
        )
    )
