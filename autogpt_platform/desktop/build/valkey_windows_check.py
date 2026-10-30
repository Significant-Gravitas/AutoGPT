"""Require the pinned Valkey version through a native Windows launch."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from artifacts import VALKEY_VERSION

WINDOWS = sys.platform == "win32"
TIMEOUT = 15


def _text(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return (value or "")[:32768]


def check(output: Path) -> dict:
    report = {"ok": False, "expected_version": VALKEY_VERSION}
    try:
        if not WINDOWS:
            raise RuntimeError("The Valkey gate requires native Windows Python")
        server = output / "valkey-server.exe"
        for required in (server, output / "msys-2.0.dll"):
            if not required.is_file() or required.stat().st_size == 0:
                raise FileNotFoundError(f"Missing or empty Valkey binary: {required}")
        report["server_sha256"] = hashlib.sha256(server.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory(prefix="valkey-native-") as temporary:
            probe = Path(temporary)
            # Checker dispatch exits even when arguments are lost; server dispatch
            # could instead start a listener with the default configuration.
            checker = probe / "valkey-check-rdb.exe"
            shutil.copy2(server, checker)
            for library in output.glob("*.dll"):
                shutil.copy2(library, probe / library.name)
            report["checker_sha256"] = hashlib.sha256(checker.read_bytes()).hexdigest()
            if report["checker_sha256"] != report["server_sha256"]:
                raise RuntimeError("Valkey checker copy differs from the server")
            report.update(executable=str(checker), cwd=str(probe), args=["--version"])
            previous_mode = ctypes.windll.kernel32.SetErrorMode(
                0x0001 | 0x0002 | 0x8000
            )
            try:
                result = subprocess.run(
                    f'"{checker}" --version',
                    executable=str(checker),
                    shell=False,
                    cwd=probe,
                    env={**os.environ, "PATH": str(probe)},
                    capture_output=True,
                    text=True,
                    errors="replace",
                    timeout=TIMEOUT,
                    check=False,
                )
            finally:
                ctypes.windll.kernel32.SetErrorMode(previous_mode)
            report.update(
                returncode=result.returncode,
                windows_status=f"0x{result.returncode & 0xffffffff:08X}",
                stdout=_text(result.stdout),
                stderr=_text(result.stderr),
            )
            if result.returncode != 0:
                raise RuntimeError("Valkey native version check exited unsuccessfully")
            expected = (
                rf"valkey-check-rdb v={re.escape(VALKEY_VERSION)} "
                r"sha=[0-9a-f]+:\d+ malloc=\S+ bits=64 build=[0-9a-f]+"
            )
            if (
                not re.fullmatch(expected, result.stdout.strip())
                or result.stderr.strip()
            ):
                raise RuntimeError("Valkey did not report the exact pinned version")
        report["ok"] = True
    except subprocess.TimeoutExpired as error:
        report.update(
            error=str(error),
            timeout_seconds=TIMEOUT,
            stdout=_text(error.stdout),
            stderr=_text(error.stderr),
        )
    except OSError as error:
        report.update(
            error=str(error),
            winerror=getattr(error, "winerror", None),
            errno=error.errno,
        )
    except RuntimeError as error:
        report["error"] = str(error)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    output = parser.parse_args(argv).output.resolve()
    diagnostics = output.with_name(output.name + "-diagnostics")
    diagnostics.mkdir(parents=True, exist_ok=True)
    report = check(output)
    text = json.dumps(report, indent=2) + "\n"
    (diagnostics / "native-launch.json").write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
