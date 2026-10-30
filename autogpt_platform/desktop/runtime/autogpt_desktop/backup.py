from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import sys
import tempfile
import uuid
import zipfile
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from types import TracebackType

from autogpt_desktop import install, migrations
from autogpt_desktop.layout import Bundle, DataDir
from autogpt_desktop.maintenance import (
    DataLock,
    MaintenanceOperation,
    directory_identity,
    sync_directory,
)
from autogpt_desktop.process import run_tool

FORMAT = 1
MANIFEST = "backup.json"
CHUNK = 1024 * 1024
RESERVE = 64 * 1024 * 1024
EXCLUDED = {"run/children.json", "run/children.json.new"}


class BackupError(RuntimeError):
    pass


def require_inactive(data: DataDir) -> None:
    if any(
        os.path.lexists(data.root / name)
        for name in ("SingletonLock", "SingletonSocket", "SingletonCookie")
    ):
        raise BackupError(
            "The desktop browser profile is still locked. Quit AutoGPT completely before continuing; if the lock is stale, start and quit the current app version first."
        )
    if (data.postgres / "postmaster.pid").exists():
        raise BackupError(
            "The database is running or did not stop cleanly. Start AutoGPT, then quit it fully before backing up or restoring."
        )
    children = data.run / "children.json"
    if children.exists():
        import psutil

        try:
            entries = json.loads(children.read_text("utf-8"))
            for entry in entries:
                try:
                    process = psutil.Process(entry["pid"])
                    if abs(process.create_time() - entry["started"]) <= 1:
                        raise BackupError(
                            "An AutoGPT service is still running. Quit AutoGPT completely first."
                        )
                except psutil.NoSuchProcess:
                    continue
        except (ValueError, TypeError, KeyError, psutil.AccessDenied) as exc:
            raise BackupError(
                "Cannot verify that AutoGPT's services have stopped."
            ) from exc


def require_stopped(bundle: Bundle, data: DataDir) -> str:
    require_inactive(data)
    if (data.config / "first-run.incomplete").exists():
        raise BackupError(
            "The first database setup is incomplete. Finish setup before backing up."
        )
    try:
        major = (data.postgres / "PG_VERSION").read_text("ascii").strip()
    except OSError as exc:
        raise BackupError("This is not an initialized AutoGPT data folder.") from exc
    result = run_tool(
        [str(bundle.postgres_bin("pg_controldata")), str(data.postgres)],
        capture_output=True,
        timeout=30,
        env={**os.environ, "LC_ALL": "C", "LANG": "C"},
    )
    if result.returncode or not re.search(
        r"^Database cluster state:\s+shut down\s*$",
        result.stdout.decode(errors="replace"),
        re.M,
    ):
        raise BackupError(
            "PostgreSQL does not report a clean shutdown. Start AutoGPT with its current version and quit it before continuing."
        )
    return major


def require_postgres(bundle: Bundle, major: str) -> None:
    result = run_tool(
        [str(bundle.postgres_bin("postgres")), "--version"],
        capture_output=True,
        timeout=30,
    )
    match = re.search(r"\b(\d+)\.\d+", result.stdout.decode(errors="replace"))
    if result.returncode or not match or match.group(1) != major:
        raise BackupError(
            "This backup requires the same PostgreSQL major version as the app that created it."
        )


def install_name(root: Path) -> str:
    stamp = root / "autogpt-install.json"
    if not stamp.exists():
        return install.NORMAL
    try:
        name = json.loads(stamp.read_text("utf-8"))["install"]
        if not isinstance(name, str) or not install.NAME.fullmatch(name):
            raise ValueError("invalid install identity")
        return name
    except (ValueError, KeyError, TypeError) as exc:
        raise BackupError("The data folder's install identity is invalid.") from exc


def inventory(root: Path) -> dict[str, tuple[int, int, int]]:
    found = {}
    for directory, folders, files in os.walk(root, followlinks=False):
        for name in sorted([*folders, *files]):
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            info = path.lstat()
            if (
                stat.S_ISLNK(info.st_mode)
                or getattr(info, "st_file_attributes", 0) & 0x400
            ):
                raise BackupError(
                    f"The data folder contains a link or junction ({relative}). Backup refuses to follow it or silently omit its contents."
                )
            if relative in EXCLUDED:
                continue
            if stat.S_ISDIR(info.st_mode):
                found[relative + "/"] = (0, info.st_mtime_ns, info.st_mode)
            elif stat.S_ISREG(info.st_mode):
                found[relative] = (info.st_size, info.st_mtime_ns, info.st_mode)
            else:
                raise BackupError(
                    f"The data folder contains an unsupported special file ({relative})."
                )
    return found


def require_space(directory: Path, size: int) -> None:
    if shutil.disk_usage(directory).free < size + RESERVE:
        raise BackupError(
            f"Not enough free space: allow at least {size + RESERVE:,} bytes without deleting the current data or previous backups."
        )


def create_backup(
    bundle: Bundle, directory: Path, destination: Path, identity: str
) -> Path:
    with DataLock(directory) as held, MaintenanceOperation(held):
        root = held.root
        target = destination.resolve()
        if target.is_relative_to(root):
            raise BackupError("Choose a backup file outside the AutoGPT data folder.")
        if target.exists():
            raise BackupError(
                "The backup destination already exists; choose a new filename."
            )
        if install_name(root) != identity:
            raise BackupError(
                "This data belongs to a different AutoGPT install or variant."
            )
        data = DataDir(root)
        major = require_stopped(bundle, data)
        require_postgres(bundle, major)
        history = migrations.bundled_history(bundle)
        before = inventory(root)
        require_space(target.parent, sum(value[0] for value in before.values()))
        descriptor, temporary = tempfile.mkstemp(
            prefix=".autogpt-backup-", dir=target.parent
        )
        pending = Path(temporary)
        try:
            entries = {}
            with os.fdopen(descriptor, "w+b") as output, zipfile.ZipFile(
                output, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True
            ) as archive:
                for name, (size, _, mode) in before.items():
                    if name.endswith("/"):
                        entries[name] = {
                            "size": 0,
                            "mode": mode & 0o700,
                            "sha256": None,
                        }
                        continue
                    digest = hashlib.sha256()
                    with (root / name).open("rb") as source, archive.open(
                        "data/" + name, "w", force_zip64=True
                    ) as saved:
                        while content := source.read(CHUNK):
                            digest.update(content)
                            saved.write(content)
                    entries[name] = {
                        "size": size,
                        "mode": mode & 0o700,
                        "sha256": digest.hexdigest(),
                    }
                if inventory(root) != before:
                    raise BackupError(
                        "The data changed while it was being backed up; no backup was published."
                    )
                require_stopped(bundle, data)
                manifest = {
                    "format": FORMAT,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "platform": sys.platform,
                    "machine": platform.machine().lower(),
                    "install": identity,
                    "source_directory": str(root),
                    "postgres_major": major,
                    "bundle_migrations": {
                        name: sorted(values) for name, values in history.items()
                    },
                    "files": entries,
                }
                archive.writestr(MANIFEST, json.dumps(manifest, sort_keys=True))
            with pending.open("r+b") as complete:
                os.fsync(complete.fileno())
            os.link(pending, target)
            sync_directory(target.parent)
            return target
        finally:
            pending.unlink(missing_ok=True)


def read_manifest(archive: zipfile.ZipFile) -> dict:
    names = archive.namelist()
    if len(names) != len(set(names)) or names.count(MANIFEST) != 1:
        raise BackupError("The backup has duplicate entries or no unique manifest.")
    if archive.getinfo(MANIFEST).file_size > 16 * 1024 * 1024:
        raise BackupError("The backup manifest is too large.")
    manifest = json.loads(archive.read(MANIFEST))
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise BackupError("This backup format is not supported.")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise BackupError("The backup file inventory is missing.")
    folded = set()
    expected = {MANIFEST}
    for name, entry in files.items():
        clean = name.rstrip("/")
        path = PurePosixPath(clean)
        if (
            not clean
            or str(path) != clean
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in name
            or ":" in name
        ):
            raise BackupError("The backup contains an unsafe path.")
        if any(
            part.endswith((".", " "))
            or re.fullmatch(r"(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?", part)
            for part in path.parts
        ):
            raise BackupError(
                "The backup contains a path that is not safe on supported systems."
            )
        key = clean.casefold() if sys.platform == "win32" else clean
        if key in folded:
            raise BackupError("The backup contains colliding paths.")
        folded.add(key)
        if (
            not isinstance(entry, dict)
            or type(entry.get("size")) is not int
            or entry["size"] < 0
            or type(entry.get("mode")) is not int
        ):
            raise BackupError("The backup inventory contains invalid file metadata.")
        if not name.endswith("/"):
            if not isinstance(entry.get("sha256"), str) or not re.fullmatch(
                "[0-9a-f]{64}", entry["sha256"]
            ):
                raise BackupError("The backup inventory contains an invalid checksum.")
            member = "data/" + name
            expected.add(member)
            if (
                member not in names
                or archive.getinfo(member).file_size != entry["size"]
            ):
                raise BackupError(
                    "The backup is incomplete or has the wrong file sizes."
                )
    if set(names) != expected:
        raise BackupError("The backup contains files outside its inventory.")
    return manifest


def publish_restore(held: DataLock, stage: Path) -> Path | None:
    root = held.root
    original = root.stat() if root.exists() else None
    previous = (
        root.with_name(f"{root.name}.before-restore-{uuid.uuid4().hex[:12]}")
        if original is not None
        else None
    )
    journal = {
        "format": FORMAT,
        "directory": str(root),
        "stage": str(stage),
        "previous": str(previous) if previous else None,
    }
    descriptor = os.open(held.journal, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        json.dump(journal, output)
        output.flush()
        os.fsync(output.fileno())
    sync_directory(root.parent)
    moved_previous = False
    published = False
    settled = False
    try:
        if previous is not None:
            require_inactive(DataDir(root))
            root.rename(previous)
            moved_previous = True
            sync_directory(root.parent)
        stage.rename(root)
        published = True
        sync_directory(root.parent)
        settled = True
        return previous
    except Exception as exc:
        if moved_previous and previous is not None and not root.exists():
            try:
                previous.rename(root)
                sync_directory(root.parent)
                settled = True
            except OSError:
                settled = False
        elif not moved_previous and not published:
            current = root.stat() if root.exists() else None
            settled = (original is None and current is None) or (
                original is not None
                and current is not None
                and (original.st_dev, original.st_ino)
                == (current.st_dev, current.st_ino)
            )
        if not settled:
            raise BackupError(
                f"Restore needs recovery. Keep all folders: current {root}, previous {previous}, staged {stage}. "
                f"The journal {held.journal} blocks AutoGPT from initializing or changing this data."
            ) from exc
        raise
    finally:
        if settled:
            held.journal.unlink()
            sync_directory(root.parent)


def _remove_readonly_file(
    remove: Callable[[str], object],
    path: str,
    exc_info: tuple[type[BaseException], BaseException, TracebackType | None],
) -> None:
    error = exc_info[1]
    if (
        sys.platform != "win32"
        or not isinstance(error, PermissionError)
        or remove not in (os.unlink, os.remove)
    ):
        raise error
    info = os.lstat(path)
    attributes = getattr(info, "st_file_attributes", 0)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT
        or not attributes & stat.FILE_ATTRIBUTE_READONLY
    ):
        raise error
    os.chmod(path, info.st_mode | stat.S_IWUSR)
    remove(path)


def restore_backup(
    bundle: Bundle, directory: Path, source: Path, identity: str
) -> Path | None:
    with DataLock(directory) as held, MaintenanceOperation(held):
        root = held.root
        if source.resolve().is_relative_to(root):
            raise BackupError(
                "Move the backup outside the data folder before restoring it."
            )
        if root.exists():
            if (
                not (root / "autogpt-install.json").is_file()
                and not (root / "postgres" / "PG_VERSION").is_file()
            ):
                raise BackupError(
                    "The destination exists but is not an AutoGPT data folder."
                )
            if install_name(root) != identity:
                raise BackupError(
                    "The current data belongs to another install or variant."
                )
            require_inactive(DataDir(root))
        with zipfile.ZipFile(source) as archive:
            manifest = read_manifest(archive)
            if (
                manifest.get("platform") != sys.platform
                or manifest.get("machine") != platform.machine().lower()
            ):
                raise BackupError(
                    "A physical database backup can only be restored on the same operating system and architecture."
                )
            original = manifest.get("source_directory")
            if (
                not isinstance(original, str)
                or manifest.get("install") != identity
                or directory_identity(Path(original)) != directory_identity(root)
            ):
                raise BackupError(
                    "Restore this backup to its original data folder and install identity."
                )
            require_postgres(bundle, manifest.get("postgres_major", ""))
            known = migrations.bundled_history(bundle)
            saved = manifest.get("bundle_migrations")
            if not isinstance(saved, dict) or any(
                not isinstance(values, list)
                or not values
                or any(
                    not isinstance(value, str) or value not in known.get(name, set())
                    for value in values
                )
                for name, values in saved.items()
            ):
                raise BackupError(
                    "This app lacks or changes migrations required by the backup. Use the app version that created the backup, or a compatible newer version."
                )
            entries = manifest["files"]
            require_space(root.parent, sum(value["size"] for value in entries.values()))
            stage = Path(
                tempfile.mkdtemp(prefix=f".{root.name}.restore-", dir=root.parent)
            )
            try:
                for name, entry in entries.items():
                    target = stage / name
                    if name.endswith("/"):
                        target.mkdir(mode=0o700, parents=True, exist_ok=True)
                        continue
                    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                    digest = hashlib.sha256()
                    with archive.open("data/" + name) as stream, target.open(
                        "xb"
                    ) as output:
                        while content := stream.read(CHUNK):
                            digest.update(content)
                            output.write(content)
                        output.flush()
                        os.fsync(output.fileno())
                    target.chmod(entry["mode"] & 0o700)
                    if digest.hexdigest() != entry["sha256"]:
                        raise BackupError(
                            "A backup checksum does not match. The existing data was not replaced."
                        )
                if install_name(stage) != identity:
                    raise BackupError(
                        "The backed-up install identity does not match the manifest."
                    )
                if require_stopped(bundle, DataDir(stage)) != manifest.get(
                    "postgres_major"
                ):
                    raise BackupError(
                        "The backed-up database version does not match its manifest."
                    )
                for directory, _, _ in os.walk(stage, topdown=False):
                    sync_directory(Path(directory))
                return publish_restore(held, stage)
            finally:
                if (
                    stage.exists()
                    and not os.path.lexists(held.journal)
                    and stage.parent == root.parent
                    and stage.name.startswith(f".{root.name}.restore-")
                ):
                    original_error = sys.exception()
                    try:
                        shutil.rmtree(stage, onerror=_remove_readonly_file)
                    except OSError:
                        if original_error is None:
                            raise
                        original_error.add_note(
                            f"Restore staging cleanup also failed. Temporary data may remain at {stage}."
                        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Back up or restore the complete stopped AutoGPT data folder. Backups contain private credentials and encryption keys; keep them private."
    )
    parser.add_argument("command", choices=("create", "restore"))
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--file", required=True, type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--install", default=install.name())
    parser.add_argument("--confirm-restore", action="store_true")
    args = parser.parse_args()
    if args.command == "restore" and not args.confirm_restore:
        parser.error(
            "restore requires --confirm-restore; the current data will be retained beside the restored folder"
        )
    bundle = Bundle(args.runtime) if args.runtime else Bundle.locate()
    try:
        if args.command == "create":
            result = create_backup(bundle, args.data_dir, args.file, args.install)
            print(f"Backup created: {result}")
        else:
            previous = restore_backup(bundle, args.data_dir, args.file, args.install)
            print(
                "Backup restored. Start AutoGPT to check the account, saved work and credentials."
            )
            if previous:
                print(f"Previous data retained: {previous}")
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
        print(f"Backup/restore stopped: {exc}", file=sys.stderr)
        for note in getattr(exc, "__notes__", ()):
            print(note, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
