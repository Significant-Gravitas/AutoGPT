from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
import unicodedata
from pathlib import Path


class DataBusyError(RuntimeError):
    pass


def directory_identity(directory: Path) -> str:
    name = str(directory.resolve())
    if os.name == "nt":
        if name[:8].lower() == "\\\\?\\unc\\":
            name = "\\\\" + name[8:]
        elif name.startswith("\\\\?\\"):
            name = name[4:]
    name = os.path.normcase(name)
    if sys.platform == "darwin":
        name = unicodedata.normalize("NFC", name).lower()
    return name


def sync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def process_started(pid: int) -> float | None:
    import psutil

    if type(pid) is not int or not 0 < pid <= 2**31 - 1:
        raise ValueError("Invalid process identity")
    try:
        started = psutil.Process(pid).create_time()
    except psutil.NoSuchProcess:
        return None
    if not math.isfinite(started) or started <= 0:
        raise ValueError("Invalid process birth time")
    return started


def active_owner(path: Path) -> bool:
    import psutil

    try:
        if path.is_symlink():
            return True
        owner = json.loads(path.read_text("utf-8"))
        pid = owner.get("pid") if isinstance(owner, dict) else None
        if type(pid) is not int or not 0 < pid <= 2**31 - 1:
            return True
        if not psutil.pid_exists(pid):
            return False
        started = owner.get("started")
        if (
            type(started) not in (int, float)
            or not math.isfinite(started)
            or started <= 0
        ):
            return True
        return process_started(pid) == started
    except (OSError, ValueError, OverflowError, psutil.Error):
        return True


class DataLock:
    def __init__(self, directory: Path):
        self.root = directory.resolve()
        self.key = hashlib.sha256(directory_identity(self.root).encode()).hexdigest()[
            :24
        ]
        self.path = self.root.parent / f".autogpt-maintenance-{self.key}.lock"
        self.journal = self.root.parent / f".autogpt-restore-{self.key}.json"
        self.descriptor: int | None = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink():
            raise DataBusyError(
                "The AutoGPT maintenance lock is a link; refusing to use it."
            )
        descriptor = os.open(
            self.path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600
        )
        try:
            if os.name == "nt":
                import msvcrt

                if os.fstat(descriptor).st_size == 0:
                    os.write(descriptor, b"\0")
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(descriptor)
            raise DataBusyError(
                "AutoGPT or a backup/restore operation is using this data folder. "
                "Quit AutoGPT completely and wait for it to finish."
            ) from exc
        self.descriptor = descriptor
        if os.path.lexists(self.journal):
            self.__exit__()
            raise DataBusyError(
                f"A restore needs recovery before AutoGPT can use this data folder. "
                f"Keep the current and retained folders; recovery paths are recorded in {self.journal}."
            )
        return self

    def __exit__(self, *exc):
        if self.descriptor is not None:
            os.close(self.descriptor)
            self.descriptor = None


class MaintenanceOperation:
    def __init__(self, held: DataLock):
        self.held = held
        self.path = held.root.parent / f".autogpt-operation-{held.key}.json"
        self.signature: tuple[int, int] | None = None

    def __enter__(self):
        if os.path.lexists(self.path) and active_owner(self.path):
            raise DataBusyError(
                "Another maintenance operation is active or its owner cannot be verified."
            )
        started = process_started(os.getpid())
        if started is None:
            raise DataBusyError("The maintenance process identity cannot be verified.")
        descriptor, name = tempfile.mkstemp(
            prefix=".autogpt-operation-pending-", dir=self.path.parent
        )
        pending = Path(name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump({"pid": os.getpid(), "started": started}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(pending, self.path)
            info = self.path.stat()
            self.signature = (info.st_dev, info.st_ino)
            sync_directory(self.path.parent)
            leases = self.path.parent.glob(f".autogpt-shell-{self.held.key}-*.json")
            if any(active_owner(lease) for lease in leases):
                raise DataBusyError(
                    "The AutoGPT desktop shell is still running or its lease cannot be verified. Quit AutoGPT completely before backing up or restoring."
                )
            return self
        except BaseException:
            self.__exit__()
            raise
        finally:
            pending.unlink(missing_ok=True)

    def __exit__(self, *exc):
        if self.signature is not None:
            try:
                info = self.path.lstat()
                if (info.st_dev, info.st_ino) == self.signature:
                    self.path.unlink()
                    sync_directory(self.path.parent)
            except FileNotFoundError:
                pass
            self.signature = None
