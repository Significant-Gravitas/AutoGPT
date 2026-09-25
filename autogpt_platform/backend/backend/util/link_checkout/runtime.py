"""The broker's local runtime: one private browser and checkout state per chat.

Everything lives under ``/dev/shm`` (memory-backed), in a directory per session
key that only this OS user can read. The browser profile, the checkout record
and the "sealed" marker (``checkout_record``) share it, so retiring a payment
browser is removing a directory. The host must not swap and must not write core dumps; card numbers
pass through Chromium's memory during a payment, and ``require_runtime`` fails
closed when either could copy them to disk.
"""

import asyncio
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from pathlib import Path

from backend.util.link_checkout import ledger
from backend.util.link_checkout.checkout_record import assert_observable, has_checkout
from backend.util.link_checkout.config import https_proxy

if sys.platform == "linux":
    import fcntl
    import resource

_locked_session: ContextVar[str] = ContextVar("private_browser_lock", default="")
_OPERATION_TIMEOUT = 120
# A chat's browser profile is deleted after this long without a browser step.
# It outlasts agent-browser's own idle exit (``AGENT_BROWSER_IDLE_TIMEOUT_MS``)
# and a checkout's deadline; the cost is signing in again after an hour away.
_IDLE_SECONDS = 60 * 60
_SWEEP_INTERVAL = 5 * 60
_last_sweep = 0.0
_sweeps: set[asyncio.Task] = set()


def require_runtime() -> None:
    if sys.platform != "linux" or not shutil.which("agent-browser"):
        raise RuntimeError("Private checkout requires the Linux browser runtime")
    if Path("/sys/fs/cgroup/memory.swap.max").read_text().strip() != "0":
        raise RuntimeError("Private checkout requires memory.swap.max=0")
    if resource.getrlimit(resource.RLIMIT_CORE)[1] != 0:
        raise RuntimeError("Private checkout requires a hard core dump limit of zero")
    mounts = Path("/proc/self/mountinfo").read_text().splitlines()
    if not any(
        line.split()[4] == "/dev/shm" and " - tmpfs " in line for line in mounts
    ):
        raise RuntimeError("Private checkout requires a tmpfs at /dev/shm")


def local_runtime_ready() -> bool:
    try:
        require_runtime()
        return True
    except (OSError, RuntimeError):
        return False


def session_home(key: str) -> Path:
    require_runtime()
    root = Path(f"/dev/shm/agp-{os.getuid()}")
    root.mkdir(mode=0o700, exist_ok=True)
    directory = root / hashlib.sha256(key.encode()).hexdigest()[:32]
    directory.mkdir(mode=0o700, exist_ok=True)
    for path in (root, directory):
        stat = path.lstat()
        if path.is_symlink() or stat.st_uid != os.getuid() or stat.st_mode & 0o077:
            raise RuntimeError("Private browser directory permissions are unsafe")
    return directory


@asynccontextmanager
async def browser_operation(
    key: str, *, allow_sealed: bool = False
) -> AsyncIterator[Path]:
    """Serialize everything done to one chat's browser, across processes.

    Re-entrant within one task, so a checkout step can run browser commands.
    A sealed browser admits only status and reset operations.
    """
    directory = session_home(key)
    if _locked_session.get() == key:
        if not allow_sealed:
            assert_observable(directory)
        yield directory
        return
    _sweep_now_and_then(directory.parent)
    lock_path = directory / "operation.lock"
    with lock_path.open("a") as lock:
        async with asyncio.timeout(_OPERATION_TIMEOUT):
            while True:
                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    await asyncio.sleep(0.05)
        # Under the lock, so no concurrent step of this chat can be caught
        # between writing the ledger and writing the checkout it records.
        ledger.restore(key, directory)
        os.utime(lock_path)
        token = _locked_session.set(key)
        try:
            if not allow_sealed:
                assert_observable(directory)
            yield directory
        finally:
            _locked_session.reset(token)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def browser_env(directory: Path) -> dict[str, str]:
    engine = directory / "engine"
    engine.mkdir(mode=0o700, exist_ok=True)
    environment = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": str(engine),
        "TMPDIR": str(engine),
        "XDG_CACHE_HOME": str(engine / "cache"),
        "XDG_CONFIG_HOME": str(engine / "config"),
        "AGENT_BROWSER_EXECUTABLE_PATH": "/usr/bin/chromium",
        # Nothing reaps a private browser at turn end (a checkout spans turns);
        # the idle timeout does, and outlasts the checkout's own deadline.
        "AGENT_BROWSER_IDLE_TIMEOUT_MS": "900000",
        "AGENT_BROWSER_DOWNLOAD_PATH": str(engine / "downloads"),
        "AGENT_BROWSER_SCREENSHOT_DIR": str(engine / "screenshots"),
        "AGENT_BROWSER_SOCKET_DIR": str(directory / "socket"),
    }
    if proxy := https_proxy():
        environment["AGENT_BROWSER_PROXY"] = proxy
        environment["AGENT_BROWSER_PROXY_BYPASS"] = "<-loopback>"
        environment["AGENT_BROWSER_ARGS"] = (
            "--disable-quic,--force-webrtc-ip-handling-policy=disable_non_proxied_udp"
        )
    return environment


async def browser_command(
    directory: Path, *args: str, timeout: int = 45
) -> tuple[int, str, str]:
    # Output goes to a file in the private directory, never through a pipe
    # buffer the parent could log, and stderr is discarded.
    with tempfile.TemporaryFile(dir=directory) as output:
        process = await asyncio.create_subprocess_exec(
            "agent-browser",
            "--session",
            "browser",
            "--profile",
            str(directory / "engine" / "profile"),
            *args,
            stdout=output,
            stderr=asyncio.subprocess.DEVNULL,
            env=browser_env(directory),
            cwd=directory,
        )
        try:
            await asyncio.wait_for(process.wait(), timeout)
        except BaseException:
            if process.returncode is None:
                process.kill()
                await process.wait()
            raise
        if process.returncode:
            return 1, "", "Private browser command failed"
        output.seek(0)
        return 0, output.read(1_000_000).decode(), ""


async def browser_endpoint(directory: Path) -> str:
    rc, output, _ = await browser_command(directory, "get", "cdp-url", "--json")
    if rc:
        raise RuntimeError("No active payment browser")
    return str(json.loads(output)["data"]["cdpUrl"])


async def retire_payment_browser(key: str) -> bool:
    """Close the browser and delete its profile. False if either failed, in
    which case the chat stays sealed."""
    directory = session_home(key)
    if not (directory / "engine").exists():
        return True
    try:
        rc, _, _ = await browser_command(directory, "close", timeout=10)
        if rc:
            return False
        return await _remove_engine(directory)
    except Exception:
        return False


async def sweep_idle_browsers(root: Path, now: float) -> None:
    """Delete the profiles of chats whose browser has been idle a long time.

    They live in memory-backed ``/dev/shm``, and nothing else removes them:
    agent-browser's own idle timeout stops the processes but keeps the files.
    A chat with a checkout is left to the checkout, which retires its browser.
    One being used right now holds its lock and is skipped.
    """
    for directory in root.iterdir():
        lock_path = directory / "operation.lock"
        if (
            not (directory / "engine").exists()
            or not lock_path.exists()
            or now - lock_path.stat().st_mtime < _IDLE_SECONDS
            or has_checkout(directory)
            or (directory / "sensitive").exists()
        ):
            continue
        with lock_path.open("a") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            try:
                if list((directory / "socket").glob("*.pid")):
                    await browser_command(directory, "close", timeout=10)
                await _remove_engine(directory)
            except Exception:
                pass
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _sweep_now_and_then(root: Path) -> None:
    # In the background: closing a stale browser can take seconds, and no
    # chat's command should wait for another chat's cleanup.
    global _last_sweep
    now = time.time()
    if now - _last_sweep < _SWEEP_INTERVAL:
        return
    _last_sweep = now
    task = asyncio.create_task(_sweep_quietly(root, now))
    _sweeps.add(task)
    task.add_done_callback(_sweeps.discard)


async def _sweep_quietly(root: Path, now: float) -> None:
    try:
        await sweep_idle_browsers(root, now)
    except Exception:
        pass


async def _remove_engine(directory: Path) -> bool:
    async with asyncio.timeout(5):
        while list((directory / "socket").glob("*.pid")):
            await asyncio.sleep(0.05)
    engine = directory / "engine"
    if engine.is_symlink() or engine.resolve().parent != directory.resolve():
        return False
    shutil.rmtree(engine)
    return True
