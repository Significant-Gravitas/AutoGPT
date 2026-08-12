import asyncio
import logging
import os
import shutil
import stat
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from types import TracebackType

logger = logging.getLogger(__name__)

_HOME_PREFIX = "autogpt-codex-"
_QUARANTINE_PREFIX = ".quarantine-autogpt-codex-"
_REAPER_MARKER_PREFIX = ".reap-"
_CLEANUP_RETRY_DELAYS_SECONDS = (0.0, 0.02, 0.05, 0.1, 0.2, 0.4)
_REAPER_LOCK = threading.Lock()
_REAPER_TARGETS: set[tuple[Path, Path, Path]] = set()
_REAPER_THREAD: threading.Thread | None = None


class TemporaryCodexHomeCleanupError(RuntimeError):
    pass


class TemporaryCodexHome:
    def __init__(self, path: Path, root: Path) -> None:
        _validate_managed_target(path, root)
        self.path = path
        self.root = root
        self.auth_path = path / "auth.json"
        self.temp_path = path / "tmp"
        self.workspace_path = path / "workspace"
        self._cleaned = False

    @classmethod
    def create(cls, root: Path | None = None) -> "TemporaryCodexHome":
        resolved_root = _prepare_root(root)
        _schedule_existing_reaps(resolved_root)
        path = Path(
            tempfile.mkdtemp(
                prefix=_HOME_PREFIX,
                dir=str(resolved_root),
            )
        ).resolve()
        _validate_managed_target(path, resolved_root)
        home = cls(path, resolved_root)
        try:
            os.chmod(path, 0o700)
            home.temp_path.mkdir(mode=0o700)
            home.workspace_path.mkdir(mode=0o700)
        except BaseException:
            home.cleanup()
            raise
        return home

    def __enter__(self) -> "TemporaryCodexHome":
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.cleanup()

    async def __aenter__(self) -> "TemporaryCodexHome":
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        await asyncio.to_thread(self.cleanup)

    def cleanup(self) -> None:
        if self._cleaned:
            return
        _validate_managed_target(self.path, self.root)
        if _remove_with_retry(self.path, self.root):
            self._cleaned = True
            return

        quarantined = _quarantine(self.path, self.root)
        if quarantined is not None:
            _schedule_reap(self.root, quarantined)
            self._cleaned = True
            return

        _schedule_reap(self.root, self.path)
        raise TemporaryCodexHomeCleanupError(
            "Could not remove or quarantine isolated Codex auth home"
        )


def _prepare_root(root: Path | None) -> Path:
    candidate = (root or Path(tempfile.gettempdir()) / "autogpt-codex").absolute()
    candidate = _canonicalize_trusted_temp_base(candidate)
    for component in (candidate, *candidate.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ValueError("Codex temporary root cannot contain symbolic links")
    candidate.mkdir(mode=0o700, parents=True, exist_ok=True)
    if candidate.is_symlink():
        raise ValueError("Codex temporary root cannot contain symbolic links")
    resolved = candidate.resolve()
    if resolved == Path(resolved.anchor):
        raise ValueError("Codex temporary root cannot be a filesystem root")
    os.chmod(resolved, 0o700)
    if os.name != "nt" and stat.S_IMODE(resolved.stat().st_mode) != 0o700:
        raise PermissionError("Codex temporary root must have mode 0700")
    return resolved


def _canonicalize_trusted_temp_base(candidate: Path) -> Path:
    for base in sorted(
        _trusted_system_temp_bases(), key=lambda path: len(path.parts), reverse=True
    ):
        if not os.path.lexists(base):
            continue
        try:
            relative = candidate.relative_to(base)
        except ValueError:
            continue
        if relative == Path("."):
            return candidate
        return base.resolve() / relative
    return candidate


def _trusted_system_temp_bases() -> tuple[Path, ...]:
    if os.name == "nt":
        return ()
    return (Path("/tmp"), Path("/var"))


def _validate_managed_target(path: Path, root: Path) -> None:
    resolved_root = root.resolve()
    if resolved_root == Path(resolved_root.anchor):
        raise ValueError("Codex temporary root cannot be a filesystem root")
    if path.parent.resolve() != resolved_root:
        raise ValueError("Codex temporary home must be a direct child of its root")
    if not path.name.startswith((_HOME_PREFIX, _QUARANTINE_PREFIX)):
        raise ValueError("Refusing to manage an unexpected Codex temporary path")


def _remove_with_retry(
    path: Path,
    root: Path,
    delays: tuple[float, ...] = _CLEANUP_RETRY_DELAYS_SECONDS,
) -> bool:
    for delay in delays:
        if delay:
            time.sleep(delay)
        try:
            _remove_tree_once(path, root)
            return True
        except FileNotFoundError:
            return True
        except OSError:
            continue
    return not os.path.lexists(path)


def _remove_tree_once(path: Path, root: Path) -> None:
    _validate_managed_target(path, root)
    if not os.path.lexists(path):
        return
    if path.is_symlink():
        path.unlink()
        return
    shutil.rmtree(path, onerror=_make_writable_and_retry)


def _make_writable_and_retry(
    operation: Callable[..., object],
    path: str,
    _error: object,
) -> None:
    os.chmod(path, stat.S_IRWXU)
    operation(path)


def _quarantine(path: Path, root: Path) -> Path | None:
    _validate_managed_target(path, root)
    if not os.path.lexists(path):
        return None
    quarantined = root / f"{_QUARANTINE_PREFIX}{uuid.uuid4().hex}"
    _validate_managed_target(quarantined, root)
    try:
        os.replace(path, quarantined)
    except OSError:
        return None
    return quarantined


def _schedule_reap(root: Path, target: Path) -> None:
    _validate_managed_target(target, root)
    marker = root / f"{_REAPER_MARKER_PREFIX}{target.name}"
    descriptor = os.open(marker, os.O_CREAT | os.O_WRONLY, 0o600)
    os.close(descriptor)
    os.chmod(marker, 0o600)

    global _REAPER_THREAD
    with _REAPER_LOCK:
        _REAPER_TARGETS.add((root, target, marker))
        if _REAPER_THREAD is None or not _REAPER_THREAD.is_alive():
            _REAPER_THREAD = threading.Thread(
                target=_reaper_loop,
                name="codex-auth-home-reaper",
                daemon=True,
            )
            _REAPER_THREAD.start()


def _schedule_existing_reaps(root: Path) -> None:
    for target in root.glob(f"{_QUARANTINE_PREFIX}*"):
        if target.parent == root:
            _schedule_reap(root, target)
    for marker in root.glob(f"{_REAPER_MARKER_PREFIX}*"):
        target_name = marker.name.removeprefix(_REAPER_MARKER_PREFIX)
        if not target_name.startswith((_HOME_PREFIX, _QUARANTINE_PREFIX)):
            continue
        target = root / target_name
        if os.path.lexists(target):
            _schedule_reap(root, target)
        else:
            with suppress(OSError):
                marker.unlink(missing_ok=True)


def _reaper_loop() -> None:
    backoff_seconds = 0.05
    while True:
        with _REAPER_LOCK:
            pending = tuple(_REAPER_TARGETS)
            if not pending:
                global _REAPER_THREAD
                _REAPER_THREAD = None
                return

        removed: list[tuple[Path, Path, Path]] = []
        for root, target, marker in pending:
            if _remove_with_retry(target, root, delays=(0.0,)):
                with suppress(OSError):
                    marker.unlink(missing_ok=True)
                removed.append((root, target, marker))

        with _REAPER_LOCK:
            _REAPER_TARGETS.difference_update(removed)
        if removed:
            backoff_seconds = 0.05
            continue

        logger.warning(
            "Retrying cleanup of %d isolated Codex auth home(s)", len(pending)
        )
        time.sleep(backoff_seconds)
        backoff_seconds = min(backoff_seconds * 2, 2.0)
