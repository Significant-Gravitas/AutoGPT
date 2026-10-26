"""Exercise a same-run experimental NSIS installer on a fresh hosted Windows VM.

Only result.json is suitable for upload. private/ contains unredacted output,
profiles, database files and generated secrets; leave it to runner teardown.
The existing smoke_test.py owns the four-start checks, using its dummy model.
This driver does not prove Electron UI, ChatGPT, or OS-level network isolation.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path, PureWindowsPath

WINDOWS = sys.platform == "win32"
BUILD = Path(__file__).resolve().parent
DESKTOP = BUILD.parent
INPUTS = (
    "../backend/poetry.lock",
    "../backend/pyproject.toml",
    "../autogpt_libs/pyproject.toml",
    "../frontend/pnpm-lock.yaml",
    "../frontend/package.json",
    "package-lock.json",
    "package.json",
    "build/artifacts.py",
)
STARTS = (
    "fresh install",
    "restart",
    "restart with a password reset",
    "restart after runs were stopped",
)
INSTALL_SECONDS = 300
LIFECYCLE_SECONDS = 1800
CLEANUP_SECONDS = 120
INSTALL_GUID = str(
    uuid.uuid5(
        uuid.UUID("50e065bc-3134-11e6-9bab-38c9862bdaf3"),
        "co.agpt.autogpt.desktop.experimental",
    )
)
INSTALL_KEYS = (
    rf"Software\{INSTALL_GUID}",
    rf"Software\Microsoft\Windows\CurrentVersion\Uninstall\{INSTALL_GUID}",
)


class SmokeError(Exception):
    """A fixed diagnostic code, safe to publish without child output."""


def require(condition: bool, code: str) -> None:
    if not condition:
        raise SmokeError(code)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def windows_path(value: str) -> None:
    path = PureWindowsPath(value)
    require(
        path.is_absolute()
        and bool(re.fullmatch(r"[A-Za-z]:", path.drive))
        and not re.search(r'["<>|?*\x00-\x1f]', value)
        and ":" not in value[2:]
        and all(part not in (".", "..") for part in value.replace("\\", "/").split("/"))
        and all(not part.endswith((" ", ".")) for part in path.parts[1:]),
        "unsafe_windows_path",
    )


def plain_path(path: Path) -> Path:
    require(path.is_absolute(), "absolute_path_required")
    if WINDOWS:
        windows_path(str(path))
    for part in (path, *path.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        require(
            not stat.S_ISLNK(info.st_mode)
            and not getattr(info, "st_file_attributes", 0) & 0x400,
            "reparse_path_refused",
        )
    require(path == path.resolve(), "noncanonical_path")
    return path


def json_object(path: Path) -> dict:
    require(plain_path(path).is_file(), "missing_json_file")
    require(path.stat().st_size <= 64 * 1024, "oversized_json_file")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate_json_key")
            result[key] = value
        return result

    value = json.loads(path.read_text("utf-8"), object_pairs_hook=unique)
    require(isinstance(value, dict), "invalid_json_object")
    return value


def select_artifact(
    directory: Path, expected_commit: str, desktop: Path = DESKTOP
) -> tuple[Path, dict]:
    plain_path(directory)
    require(directory.is_dir(), "artifact_directory_missing")
    require(bool(re.fullmatch(r"[0-9a-f]{40}", expected_commit)), "invalid_commit")
    record = json_object(directory / "build-inputs.json")
    require(record.get("commit") == expected_commit, "artifact_commit_mismatch")
    require(record.get("variant") == "experimental", "artifact_variant_mismatch")
    version = record.get("version")
    require(
        isinstance(version, str)
        and bool(re.fullmatch(r"0\.0\.0-dev\.[1-9][0-9]*", version)),
        "artifact_version_mismatch",
    )
    require(
        str(record.get("runner", "")).startswith("Windows-"), "artifact_os_mismatch"
    )
    name = f"AutoGPT-experimental-Setup-{version}-x64.exe"
    require(
        {entry.name for entry in directory.iterdir()} == {"build-inputs.json", name},
        "ambiguous_artifact_contents",
    )
    installer = plain_path(directory / name)
    require(installer.is_file(), "installer_missing")
    with installer.open("rb") as stream:
        require(stream.read(2) == b"MZ", "installer_not_pe")
    hashes = record.get("sha256")
    require(
        isinstance(hashes, dict) and set(hashes) == set(INPUTS), "build_inputs_mismatch"
    )
    require(
        all(hashes[name] == sha256(desktop / name) for name in INPUTS),
        "source_inputs_mismatch",
    )
    return installer, {
        "commit": expected_commit,
        "variant": "experimental",
        "version": version,
        "installer": str(installer),
        "installer_sha256": sha256(installer),
        "build_inputs_sha256": sha256(directory / "build-inputs.json"),
        "source_inputs_sha256": hashes,
    }


def prepare_work_dir(work: Path, artifact: Path) -> Path:
    plain_path(work)
    require(not work.exists() and work.parent.is_dir(), "work_directory_must_be_new")
    for protected in (artifact, DESKTOP.parents[1]):
        require(
            not work.is_relative_to(protected) and not protected.is_relative_to(work),
            "work_directory_overlaps_input",
        )
    work.mkdir(mode=0o700)
    private = work / "private"
    private.mkdir(mode=0o700)
    for name in ("logs", "home", "appdata", "localappdata", "temp", "data"):
        (private / name).mkdir(mode=0o700)
    return private


def require_clean_host() -> None:
    require(WINDOWS, "windows_required")
    require(
        os.environ.get("GITHUB_ACTIONS") == "true"
        and os.environ.get("RUNNER_ENVIRONMENT") == "github-hosted",
        "fresh_github_hosted_runner_required",
    )
    require(
        sys.version_info >= (3, 13),
        "python_313_required_for_private_windows_directories",
    )
    import winreg

    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        for view in (winreg.KEY_WOW64_32KEY, winreg.KEY_WOW64_64KEY):
            for path in INSTALL_KEYS:
                try:
                    key = winreg.OpenKey(hive, path, 0, winreg.KEY_READ | view)
                except FileNotFoundError:
                    continue
                with key:
                    raise SmokeError("existing_experimental_install")


def short_path(path: str) -> str:
    from ctypes import wintypes

    get_short = ctypes.WinDLL("kernel32", use_last_error=True).GetShortPathNameW
    get_short.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
    get_short.restype = wintypes.DWORD
    length = get_short(path, None, 0)
    if not 0 < length <= 32768:
        return path
    buffer = ctypes.create_unicode_buffer(length)
    return buffer.value if 0 < get_short(path, buffer, length) < length else path


def require_alias_root(local: Path) -> None:
    def usable(path: str) -> bool:
        return path.isascii() and not any(char.isspace() for char in path)

    require(
        usable(str(local)) or usable(short_path(str(local))),
        "private_path_requires_space_free_short_name",
    )


def private_environment(private: Path, inherited: dict[str, str]) -> dict[str, str]:
    require_alias_root(private / "localappdata")
    inherited = {key.upper(): value for key, value in inherited.items()}
    system = plain_path(Path(inherited["SYSTEMROOT"]))
    system32 = system / "System32"
    require(
        (system32 / "WindowsPowerShell/v1.0/powershell.exe").is_file(),
        "system_powershell_missing",
    )
    env = {
        name: inherited[name]
        for name in ("SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "NUMBER_OF_PROCESSORS")
        if name in inherited
    }
    home = private / "home"
    env.update(
        {
            "PATH": os.pathsep.join(
                str(path)
                for path in (
                    system32,
                    system,
                    system32 / "Wbem",
                    system32 / "WindowsPowerShell/v1.0",
                )
            ),
            "COMSPEC": str(system32 / "cmd.exe"),
            "PATHEXT": ".COM;.EXE;.BAT;.CMD",
            "HOME": str(home),
            "USERPROFILE": str(home),
            "HOMEDRIVE": home.drive,
            "HOMEPATH": str(home)[len(home.drive) :],
            "APPDATA": str(private / "appdata"),
            "LOCALAPPDATA": str(private / "localappdata"),
            "TEMP": str(private / "temp"),
            "TMP": str(private / "temp"),
            "AUTOGPT_DESKTOP_DATA_DIR": str(private / "data"),
            "AUTOGPT_CLAUDE_CODE": "off",
            "CHAT_TEST_MODE": "true",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "DISABLE_ERROR_REPORTING": "1",
            "DISABLE_TELEMETRY": "1",
            "NEXT_TELEMETRY_DISABLED": "1",
            "OTEL_SDK_DISABLED": "true",
        }
    )
    return env


def installer_command(installer: Path, destination: Path) -> str:
    # NSIS consumes the entire unquoted remainder after /D=, including spaces.
    return subprocess.list2cmdline([str(installer), "/S", "/currentuser"]) + (
        f" /D={destination}"
    )


def installed_runtime(install: Path) -> tuple[Path, Path, dict]:
    require(plain_path(install / "autogpt-experimental.exe").is_file(), "shell_missing")
    runtime = plain_path(install / "resources/runtime")
    manifest = runtime / "manifest.json"
    require(
        json_object(manifest)
        == {
            "win32": {
                "command": "python/python.exe",
                "args": ["-B", "-m", "autogpt_desktop", "serve"],
            },
            "default": {
                "command": "python/bin/python3",
                "args": ["-B", "-m", "autogpt_desktop", "serve"],
            },
        },
        "unexpected_runtime_manifest",
    )
    python = plain_path(runtime / "python/python.exe")
    require(python.is_file(), "installed_python_missing")
    return (
        runtime,
        python,
        {
            "install": str(install),
            "runtime": str(runtime),
            "python": str(python),
            "python_sha256": sha256(python),
            "manifest_sha256": sha256(manifest),
        },
    )


def smoke_command(python: Path, runtime: Path, data: Path) -> list[str]:
    return [
        str(python),
        "-B",
        str(BUILD / "smoke_test.py"),
        str(runtime),
        "--profile",
        "compact",
        "--data-dir",
        str(data),
        "--timeout",
        "600",
    ]


def lifecycle_summary(log: Path) -> dict:
    starts = []
    success = False
    with log.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            for index, name in enumerate(STARTS, 1):
                if line.rstrip("\r\n") == f"== start {index} of 4: {name}":
                    starts.append(index)
            if re.fullmatch(r"OK in [0-9]+s\s*", line):
                success = True
    return {"starts": starts, "completed": starts == [1, 2, 3, 4] and success}


class OwnedProcesses:
    """Remember PID + birth time while parent links still exist.

    Detached processes also qualify when their executable is inside the new,
    exclusive install directory and they were born after this driver started.
    Never use names, port ownership, or command-line substrings to kill a process.
    """

    def __init__(self, install: Path, data: Path):
        import psutil

        self.psutil = psutil
        self.install = install
        self.data = data
        self.born = time.time()
        self.seen: dict[tuple[int, float], str] = {}
        self.cleanup_live: set[tuple[int, float]] | None = None
        self.ports: set[int] = set()
        self.allocated_ports: set[int] = set()
        self.inspection_failed = False

    def remember(self, process, reason: str) -> None:
        self.seen.setdefault((process.pid, process.create_time()), reason)

    def living(self) -> list:
        result = []
        for pid, born in list(self.seen):
            try:
                process = self.psutil.Process(pid)
                if process.create_time() == born and process.is_running():
                    result.append(process)
                    if self.cleanup_live is not None:
                        self.cleanup_live.add((pid, born))
            except self.psutil.NoSuchProcess:
                pass
            except self.psutil.Error:
                self.inspection_failed = True
        return result

    def observe(self) -> None:
        for process in self.living():
            try:
                for child in process.children(recursive=True):
                    self.remember(child, "observed_descendant")
            except self.psutil.NoSuchProcess:
                pass
            except self.psutil.Error:
                self.inspection_failed = True
        for process in self.psutil.process_iter():
            with contextlib.suppress(self.psutil.Error, OSError):
                if process.create_time() >= self.born:
                    executable = process.exe()
                    if executable and Path(executable).is_relative_to(self.install):
                        self.remember(process, "new_install_executable")
        for process in self.living():
            try:
                for connection in process.net_connections(kind="tcp"):
                    if connection.status == self.psutil.CONN_LISTEN:
                        self.ports.add(connection.laddr.port)
            except self.psutil.NoSuchProcess:
                pass
            except self.psutil.Error:
                self.inspection_failed = True
        self.collect_allocated_ports()

    def collect_allocated_ports(self, final: bool = False) -> None:
        path = self.data / "config/ports.json"
        if not path.exists():
            return
        try:
            ports = json_object(path)
            require(
                len(ports) <= 64
                and all(
                    type(port) is int and 0 < port < 65536 for port in ports.values()
                ),
                "invalid_allocated_ports",
            )
            self.ports.update(ports.values())
            self.allocated_ports.update(ports.values())
        except (OSError, ValueError, SmokeError):
            if final:
                self.inspection_failed = True

    def cleanup(self, timeout: float = CLEANUP_SECONDS) -> dict:
        """Cooperative budget: synchronous inspection can overrun, failing proof."""
        started = time.monotonic()
        deadline = started + timeout
        kill_deadline = deadline - min(15, max(0, timeout) / 4)
        self.cleanup_live = set()
        self.observe()
        initial = self.living()
        while time.monotonic() < kill_deadline:
            self.observe()
            living = self.living()
            if not living:
                break
            for process in reversed(living):
                if time.monotonic() >= kill_deadline:
                    break
                try:
                    process.kill()
                except self.psutil.NoSuchProcess:
                    pass
                except self.psutil.Error:
                    self.inspection_failed = True
            time.sleep(min(0.2, max(0, kill_deadline - time.monotonic())))
        self.observe()
        self.collect_allocated_ports(final=True)
        listeners = []
        unchecked = []
        if time.monotonic() >= deadline:
            unchecked = sorted(self.ports)
        else:
            live = {process.pid for process in self.living()}
            for connection in self.psutil.net_connections(kind="tcp"):
                if (
                    connection.status == self.psutil.CONN_LISTEN
                    and connection.laddr.port in self.ports
                ):
                    listeners.append(
                        {
                            "port": connection.laddr.port,
                            "pid": connection.pid,
                            "owned": connection.pid in live,
                        }
                    )
        survivors = [process.pid for process in self.living()]
        elapsed = time.monotonic() - started
        exceeded = elapsed >= timeout
        self.inspection_failed |= exceeded
        cleanup_live = self.cleanup_live
        self.cleanup_live = None
        return {
            "leftovers_before_cleanup": len(initial),
            "live_owned_during_cleanup": [
                {"pid": pid, "created": born} for pid, born in sorted(cleanup_live)
            ],
            "surviving_owned_pids": survivors,
            "observed_processes": [
                {"pid": pid, "created": born, "provenance": reason}
                for (pid, born), reason in self.seen.items()
            ],
            "observed_ports": sorted(self.ports),
            "allocated_ports": sorted(self.allocated_ports),
            "ports_still_open": sorted({entry["port"] for entry in listeners}),
            "remaining_listeners": listeners,
            "ports_not_checked": unchecked,
            "inspection_failed": self.inspection_failed,
            "elapsed_seconds": round(elapsed, 2),
            "timeout_seconds": timeout,
            "deadline_exceeded": exceeded,
        }


def run_stage(
    name: str,
    command: str | list[str],
    timeout: int,
    private: Path,
    env: dict[str, str],
    owned: OwnedProcesses,
    stages: list[dict],
) -> None:
    stage = {"name": name, "command": command, "timeout_seconds": timeout}
    stages.append(stage)
    started = time.monotonic()
    with (private / "logs" / f"{name}.log").open("wb") as output:
        process = subprocess.Popen(
            command,
            cwd=private,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            shell=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        with contextlib.suppress(owned.psutil.NoSuchProcess):
            owned.remember(owned.psutil.Process(process.pid), "launched_stage")
        try:
            while process.poll() is None:
                owned.observe()
                if time.monotonic() - started >= timeout:
                    stage["timed_out"] = True
                    raise SmokeError(f"{name}_timeout")
                time.sleep(0.25)
            stage["exit_code"] = process.returncode
            require(process.returncode == 0, f"{name}_exit")
        finally:
            stage["elapsed_seconds"] = round(time.monotonic() - started, 2)
            stage.setdefault("exit_code", process.poll())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args(argv)
    result: dict = {
        "schema": 1,
        "ok": False,
        "scope": "installed-runtime-four-start-dummy-model",
        "upload_allowlist": ["result.json"],
        "stages": [],
    }
    private = None
    owned = None
    try:
        require_clean_host()
        require(
            os.environ.get("GITHUB_SHA") == args.expected_commit, "github_sha_mismatch"
        )
        private = prepare_work_dir(args.work_dir, args.artifact_dir)
        installer, provenance = select_artifact(args.artifact_dir, args.expected_commit)
        result["provenance"] = provenance
        result["source_sha256"] = {
            name: sha256(BUILD / name)
            for name in (
                "installed_smoke_windows.py",
                "smoke_test.py",
                "../runtime/tests/backend_contract.py",
            )
        }
        env = private_environment(private, dict(os.environ))
        install = private / "install"
        owned = OwnedProcesses(install, private / "data")
        run_stage(
            "install",
            installer_command(installer, install),
            INSTALL_SECONDS,
            private,
            env,
            owned,
            result["stages"],
        )
        owned.observe()
        require(not owned.living(), "installer_left_processes")
        runtime, python, result["installed"] = installed_runtime(install)
        probe = (
            "import pathlib, sys, psutil, psycopg2, redis; "
            "root = pathlib.Path(sys.argv[1]).resolve(); "
            "assert pathlib.Path(sys.executable).resolve().is_relative_to(root); "
            "assert all(pathlib.Path(p).resolve().is_relative_to(root) "
            "for p in sys.path if p); "
            "assert all(pathlib.Path(m.__file__).resolve().is_relative_to(root) "
            "for m in (psutil, psycopg2, redis))"
        )
        run_stage(
            "runtime_imports",
            [str(python), "-B", "-c", probe, str(runtime)],
            60,
            private,
            env,
            owned,
            result["stages"],
        )
        require(not any((private / "data").iterdir()), "data_directory_not_empty")
        run_stage(
            "lifecycle",
            smoke_command(python, runtime, private / "data"),
            LIFECYCLE_SECONDS,
            private,
            env,
            owned,
            result["stages"],
        )
        result["lifecycle"] = lifecycle_summary(private / "logs/lifecycle.log")
        require(result["lifecycle"]["completed"], "incomplete_lifecycle")
        result["ok"] = True
    except SmokeError as exc:
        result["error"] = str(exc)
    except (Exception, KeyboardInterrupt):
        result["error"] = "driver_exception_private_logs_only"
        if private is not None:
            with (private / "logs/driver.log").open("w", encoding="utf-8") as log:
                traceback.print_exc(file=log)
    finally:
        if owned is not None:
            try:
                cleanup = result["cleanup"] = owned.cleanup()
                if result["ok"] and not cleanup["allocated_ports"]:
                    result["ok"] = False
                    result["error"] = "missing_allocated_ports"
                if any(
                    cleanup[name]
                    for name in (
                        "leftovers_before_cleanup",
                        "live_owned_during_cleanup",
                        "surviving_owned_pids",
                        "ports_still_open",
                        "ports_not_checked",
                        "inspection_failed",
                    )
                ):
                    result["ok"] = False
                    result.setdefault("error", "owned_process_or_port_leak")
            except Exception:
                result["ok"] = False
                result.setdefault("error", "cleanup_failed")
        if private is not None:
            (args.work_dir / "result.json").write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8"
            )
    print("Installed Windows smoke: " + ("PASS" if result["ok"] else "FAIL"))
    if not result["ok"]:
        print(result.get("error", "unknown_failure"))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
