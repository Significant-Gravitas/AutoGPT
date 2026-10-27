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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--key", action="append", dest="keys")
    parser.add_argument("--windows-host", action="append", default=["desktop"])
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    keys = args.keys or MICROSOFT_KEYS
    if any(not re.fullmatch(r"[A-Z_][A-Z0-9_]*", key) for key in keys):
        parser.error("Keys must be explicit environment variable names.")
    try:
        sources = [read_source(source, args.windows_host) for source in args.source]
        values = select_values(sources, keys)
        target = dotenv_values(args.target, interpolate=False)
        changes = {
            key: value for key, value in values.items() if target.get(key) != value
        }
        conflicts = [key for key in changes if target.get(key)]
        for key in keys:
            status = "missing at source"
            if key in values:
                status = "different" if key in changes else "matches"
            print(f"{key}: {status}")
        if not args.apply:
            print("Dry run; no files changed.")
            return
        if conflicts and not args.replace:
            raise ValueError(
                "Use --replace to replace target keys: " + ", ".join(conflicts)
            )
        if changes:
            apply_updates(args.target, changes)
        print(f"Applied {len(changes)} selected keys.")
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        # External process output can contain env values; never print it.
        message = str(error) if type(error) is ValueError else type(error).__name__
        parser.exit(2, f"Env synchronization stopped: {message}\n")


def read_source(source: str, windows_hosts: list[str]) -> dict[str, str | None]:
    if source.startswith("local:"):
        text = Path(source[6:]).read_text()
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
            text=True,
            timeout=30,
            check=True,
        )
        text = result.stdout
    return dict(dotenv_values(stream=io.StringIO(text), interpolate=False))


def windows_read_command(path: str) -> str:
    literal = "'" + path.replace("'", "''") + "'"
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
    selected = {}
    for key in keys:
        values = {source[key] for source in sources if source.get(key)}
        if len(values) > 1:
            raise ValueError(
                f"Sources disagree for {key}; choose one source explicitly."
            )
        if values:
            value = values.pop()
            if value is not None:
                selected[key] = value
    return selected


def apply_updates(target: Path, updates: dict[str, str]) -> None:
    if target.is_symlink() or target.name.endswith(
        (".default", ".example", ".template")
    ):
        raise ValueError("Target must be a private, regular env file.")
    target.parent.mkdir(parents=True, exist_ok=True)
    tracked = subprocess.run(
        ["git", "-C", str(target.parent), "ls-files", "--error-unmatch", target.name],
        capture_output=True,
        check=False,
    )
    if tracked.returncode == 0:
        raise ValueError("Refusing to write credentials to a tracked file.")
    lock = target.with_name(target.name + ".sync-lock")
    lock_fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        _write_updates(target, updates)
    finally:
        os.close(lock_fd)
        lock.unlink()


def _write_updates(target: Path, updates: dict[str, str]) -> None:
    original = target.read_bytes() if target.exists() else b""
    fd, filename = tempfile.mkstemp(prefix=".env-sync-", dir=target.parent)
    temporary = Path(filename)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(original)
        for key, value in updates.items():
            set_key(str(temporary), key, value, quote_mode="always")
        if (target.read_bytes() if target.exists() else b"") != original:
            raise ValueError("Target changed during synchronization; retry.")
        if target.exists():
            backup = target.with_name(f"{target.name}.backup-{time.time_ns()}")
            backup_fd = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(backup_fd, "wb") as stream:
                stream.write(original)
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
