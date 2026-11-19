"""Compare and pull selected env keys. No values are printed; writes are opt-in."""

import argparse
import base64
import io
import os
import re
import shlex
import subprocess
import tempfile
import time
from pathlib import Path

from dotenv import dotenv_values, set_key

MICROSOFT_KEYS = [
    "MICROSOFT_CLIENT_ID",
    "MICROSOFT_CLIENT_SECRET",
    "MICROSOFT_TENANT_ID",
]
POWERSHELL_SINGLE_QUOTES = "'\u2018\u2019\u201a\u201b"
TEMPLATE_SUFFIXES = (".default", ".example", ".template", ".sample", ".dist", ".tpl")
# Kept from the tracked check, which matches names literally: git refuses to
# combine the glob and icase settings with --literal-pathspecs.
PATHSPEC_VARIABLES = (
    "GIT_GLOB_PATHSPECS",
    "GIT_NOGLOB_PATHSPECS",
    "GIT_ICASE_PATHSPECS",
    "GIT_LITERAL_PATHSPECS",
)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    keys = args.keys or MICROSOFT_KEYS
    windows_hosts = args.windows_hosts or ["desktop"]
    if any(not re.fullmatch(r"[A-Z_][A-Z0-9_]*", key) for key in keys):
        parser.error("Keys must be explicit environment variable names.")
    try:
        check_target(args.target)
        sources = [read_source(source, windows_hosts) for source in args.source]
        values = select_values(sources, keys)
        target = dotenv_values(args.target, interpolate=False)
        for key in keys:
            print(f"{key}: {key_status(values.get(key), target.get(key))}")
        if not args.apply:
            print("Dry run; no files changed.")
            return
        applied = apply_updates(args.target, values, replace=args.replace)
        print(f"Applied {applied} selected keys.")
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        # External process output can contain env values; never print it.
        message = str(error) if type(error) is ValueError else type(error).__name__
        parser.exit(2, f"Env synchronization stopped: {message}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="an env file to read, as local:/path or ssh-host:/path; repeatable",
    )
    parser.add_argument(
        "--target", type=Path, required=True, help="the local env file to update"
    )
    parser.add_argument(
        "--key",
        action="append",
        dest="keys",
        metavar="KEY",
        help="a key to sync; repeatable (default: the three Microsoft settings)",
    )
    parser.add_argument(
        "--windows-host",
        action="append",
        dest="windows_hosts",
        metavar="HOST",
        help="an SSH host to read through PowerShell; repeatable (default: desktop)",
    )
    parser.add_argument(
        "--apply", action="store_true", help="write the keys missing at the target"
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="with --apply, also replace values that differ",
    )
    return parser


def read_source(source: str, windows_hosts: list[str]) -> dict[str, str | None]:
    if source.startswith("local:"):
        text = Path(source[6:]).read_text(encoding="utf-8")
    else:
        host, separator, path = source.partition(":")
        if not separator or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@-]*", host):
            raise ValueError("Sources must be local:/path or ssh-host:/path.")
        command = (
            windows_read_command(path)
            if host in windows_hosts
            else "cat -- " + shlex.quote(path)
        )
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, command],
            capture_output=True,
            encoding="utf-8",
            timeout=30,
            check=True,
        )
        text = result.stdout
    return dict(dotenv_values(stream=io.StringIO(text), interpolate=False))


def windows_read_command(path: str) -> str:
    # PowerShell ends a single-quoted string at a curly quote too; doubling any
    # of them escapes it, as its own EscapeSingleQuotedStringContent does.
    escaped = "".join(c * 2 if c in POWERSHELL_SINGLE_QUOTES else c for c in path)
    literal = "'" + escaped + "'"
    script = (
        "$ProgressPreference='SilentlyContinue';$ErrorActionPreference='Stop';"
        "[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false);"
        f"[Console]::Write([System.IO.File]::ReadAllText({literal}))"
    )
    encoded = base64.b64encode(script.encode("utf-16le")).decode()
    return "powershell -NoProfile -NonInteractive -EncodedCommand " + encoded


def select_values(
    sources: list[dict[str, str | None]], keys: list[str]
) -> dict[str, str]:
    selected: dict[str, str] = {}
    for key in keys:
        values = {value for source in sources if (value := source.get(key))}
        if len(values) > 1:
            raise ValueError(
                f"Sources disagree for {key}; choose one source explicitly."
            )
        if values:
            selected[key] = values.pop()
    return selected


def key_status(source: str | None, target: str | None) -> str:
    if source is None:
        return "missing at source"
    if target == source:
        return "matches"
    return "different" if target else "missing at target"


def apply_updates(target: Path, values: dict[str, str], replace: bool = False) -> int:
    """Write the selected values that differ; return how many were written.

    What changes, and whether that replaces a value, is decided from the
    target as read under its lock, so a value another run wrote after this
    run's report still needs ``replace``.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    check_target(target)
    lock = target.with_name(target.name + ".sync-lock")
    try:
        lock_fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise ValueError(
            f"{lock.name} exists: another sync is running or one was killed. "
            "Delete it if none is running."
        ) from None
    try:
        return _write_updates(target, values, replace)
    finally:
        os.close(lock_fd)
        lock.unlink()


def check_target(target: Path) -> None:
    """Refuse a target that must not hold credentials, before reading anything."""
    if target.is_symlink() or target.name.lower().endswith(TEMPLATE_SUFFIXES):
        raise ValueError("Target must be a private, regular env file.")
    # Only a dry run meets a missing directory; --apply creates it first.
    if target.parent.is_dir() and _tracked(target):
        raise ValueError("Refusing to write credentials to a tracked file.")


def _tracked(target: Path) -> bool:
    """Whether the repository around *target* tracks it.

    Git's repository variables (``GIT_DIR``, ``GIT_INDEX_FILE`` and the rest)
    are dropped so an inherited one cannot point the check at another
    repository, and the name is matched literally, never as pathspec magic
    such as ``:(literal)``.  Any answer but tracked, untracked or "not a git
    repository" stops the sync.
    """
    local = subprocess.run(
        ["git", "rev-parse", "--local-env-vars"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    dropped = {*local, *PATHSPEC_VARIABLES}
    env = {name: value for name, value in os.environ.items() if name not in dropped}
    ls_files = ["git", "--literal-pathspecs", "-C", str(target.parent), "ls-files"]
    result = subprocess.run(
        [*ls_files, "--error-unmatch", "--", target.name],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env={**env, "LC_ALL": "C"},
        check=False,
    )
    if result.returncode in (0, 1):
        return result.returncode == 0
    if result.returncode == 128 and "not a git repository" in result.stderr:
        return False
    raise ValueError("Could not tell whether git tracks the target.")


def _write_updates(target: Path, values: dict[str, str], replace: bool) -> int:
    original = target.read_bytes() if target.exists() else b""
    current = dotenv_values(
        stream=io.StringIO(original.decode("utf-8")), interpolate=False
    )
    changes = {key: value for key, value in values.items() if current.get(key) != value}
    conflicts = [key for key in changes if current.get(key)]
    if conflicts and not replace:
        raise ValueError(
            "Use --replace to replace target keys: " + ", ".join(conflicts)
        )
    if not changes:
        return 0
    updated = _edited(original, changes)
    if (target.read_bytes() if target.exists() else b"") != original:
        raise ValueError("Target changed during synchronization; retry.")
    if target.exists():
        print(f"Saved the previous {target.name} to {back_up(target, original)}")
    # The replacement has to sit beside the target for an atomic replace.
    fd, filename = tempfile.mkstemp(prefix=".env-sync-", dir=target.parent)
    temporary = Path(filename)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(updated)
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return len(changes)


def _edited(original: bytes, changes: dict[str, str]) -> bytes:
    """The target's bytes with *changes* applied, drafted in the private directory.

    ``set_key`` rewrites a file through a temporary file beside it, which here
    lands outside the target's repository.
    """
    with tempfile.TemporaryDirectory(dir=private_directory()) as scratch:
        draft = Path(scratch) / "draft.env"
        draft.write_bytes(original)
        for key, value in changes.items():
            set_key(str(draft), key, value, quote_mode="always")
        return draft.read_bytes()


def back_up(target: Path, original: bytes) -> Path:
    """Copy the target into a private directory outside any repository.

    The copy holds every secret in the target, so beside the target ``git add``
    could stage it.
    """
    directory = private_directory()
    name = f"{target.parent.name}-{target.name.lstrip('.')}-{time.time_ns()}"
    backup_fd = os.open(directory / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(backup_fd, "wb") as stream:
        stream.write(original)
    return directory / name


def private_directory() -> Path:
    directory = backup_directory()
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    return directory


def backup_directory() -> Path:
    state = os.environ.get("XDG_STATE_HOME", "")
    root = Path(state) if os.path.isabs(state) else Path.home() / ".local" / "state"
    return root / "autogpt-env-sync"


if __name__ == "__main__":
    main()
