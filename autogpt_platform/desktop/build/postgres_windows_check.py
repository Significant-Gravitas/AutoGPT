"""Hosted Windows component gate. Only work-dir/result.json is upload-safe.

The private tree contains synthetic credentials, native output and a database.
This checks the pinned component and source initializer, not an installed app.
The legacy ACL comparison is diagnostic only and never passes this gate.
"""

from __future__ import annotations

import argparse
import ctypes
import gzip
import hashlib
import json
import math
import os
import re
import secrets
import subprocess
import sys
import tarfile
import time
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath, PureWindowsPath

BUILD = Path(__file__).resolve().parent
sys.path.insert(0, str(BUILD))

import installed_smoke_windows as smoke
import windows_private_acl as acl
from artifacts import ARTIFACTS

DESKTOP = BUILD.parent
PIN = "920bebd11a2cc336096b3df382ed02c21df050d18a25420bf9049e50dd5bb38e"  # pragma: allowlist secret
URL = (
    "https://github.com/boomship/postgres-vector-embedded/releases/download/"
    "v0.3.1/postgres-lite-win32-x64.tar.gz"
)
MAX_ARCHIVE = 64 * 1024 * 1024
MAX_EXPANDED = 128 * 1024 * 1024
MAX_OUTPUT = 128 * 1024
FETCH_SECONDS = 120
VERSION_SECONDS = 5
INITIALIZE_SECONDS = 60
WORKER_SECONDS = 75
NATIVE_SECONDS = 90
CLEANUP_SECONDS = 10
SOURCES = {
    "probe": "build/postgres_windows_check.py",
    "private_acl": "build/windows_private_acl.py",
    "artifacts": "build/artifacts.py",
    "installed_driver": "build/installed_smoke_windows.py",
    "runtime_package": "runtime/autogpt_desktop/__init__.py",
    "postgres": "runtime/autogpt_desktop/postgres.py",
    "layout": "runtime/autogpt_desktop/layout.py",
    "process": "runtime/autogpt_desktop/process.py",
    "winlinks": "runtime/autogpt_desktop/winlinks.py",
}
STAGES = {"host", "artifact", "versions", "initialize", "verify", "cleanup", "complete"}
STATUSES = {"not_run", "unknown", "passed", "failed", "timeout", "error"}
CLASSES = {
    "restricted_reexec",
    "sibling_backend",
    "share_input",
    "password_read",
    "data_directory",
    "bootstrap",
    "unknown",
}


def require(condition: bool) -> None:
    if not condition:
        raise ValueError("component check failed")


def integer(value, lower=-2147483648, upper=4294967295):
    require(type(value) is int and lower <= value <= upper)
    return value


def empty_native() -> dict:
    return {
        "status": "not_run",
        "child_exit_code": None,
        "windows_status": None,
        "windows_error": None,
        "classification": "unknown",
        "output_rejected": False,
        "capture_failed": False,
    }


def classify_output(stdout: bytes, stderr: bytes, password_file: str = "") -> dict:
    """Hints from REL_18_6 initdb.c and common/restricted_token.c only."""
    result = {
        "classification": "unknown",
        "windows_error": None,
        "output_rejected": False,
    }
    try:
        require(type(stdout) is bytes and type(stderr) is bytes)
        require(len(stdout) + len(stderr) <= MAX_OUTPUT)
        texts = [raw.decode("utf-8") for raw in (stdout, stderr)]
        lines = [line for text in texts for line in text.splitlines()]
        require(len(lines) <= 2048)
        require(all(len(line.encode("utf-8")) <= 4096 for line in lines))
        require(
            not any(
                re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", text) for text in texts
            )
        )
        found = set()
        errors = set()
        for line in lines:
            match = re.fullmatch(
                r"initdb: error: could not (?:open process token|allocate SIDs|"
                r"create restricted token|re-execute with restricted token|"
                r'start process for command "[^"\r\n]+"): error code ([0-9]{1,10})',
                line,
            )
            if match:
                found.add("restricted_reexec")
                errors.add(integer(int(match[1]), 0))
            if re.fullmatch(
                r'initdb: error: program "postgres" (?:is needed by initdb but was not '
                r'found in the same directory as "[^"\r\n]+"|was found by '
                r'"[^"\r\n]+" but was not the same version as initdb)',
                line,
            ):
                found.add("sibling_backend")
            if re.fullmatch(
                r'initdb: error: (?:could not read password from file "[^"\r\n]+": .+|'
                r'password file "[^"\r\n]+" is empty)',
                line,
            ) or (
                password_file
                and re.fullmatch(
                    r'initdb: error: could not open file "'
                    + re.escape(password_file)
                    + r'" for reading: .+',
                    line,
                )
            ):
                found.add("password_read")
            if re.fullmatch(
                r"initdb: error: (?:could not (?:create|access|change permissions of) "
                r'directory "[^"\r\n]+": .+|directory "[^"\r\n]+" exists but is not empty)',
                line,
            ):
                found.add("data_directory")
            if line == (
                "initdb: hint: This might mean you have a corrupted installation or "
                "identified the wrong directory with the invocation option -L."
            ):
                found.add("share_input")
            if re.fullmatch(
                r'initdb: error: input file "[^"\r\n]+" does not belong to PostgreSQL 18\.6',
                line,
            ):
                found.add("share_input")
            if re.fullmatch(
                r"(?:running bootstrap script|performing post-bootstrap initialization) "
                r"\.\.\. could not write to child process: .+",
                line,
            ):
                found.add("bootstrap")
        if len(found) == 1:
            result["classification"] = found.pop()
        if len(errors) == 1:
            result["windows_error"] = errors.pop()
    except (ValueError, TypeError, UnicodeError, OverflowError):
        result.update(
            classification="unknown", windows_error=None, output_rejected=True
        )
    return result


def native_result(code: int, stdout: bytes, stderr: bytes, password_file="") -> dict:
    result = empty_native()
    integer(code)
    result.update(classify_output(stdout, stderr, password_file))
    result.update(child_exit_code=code, windows_status=code & 0xFFFFFFFF)
    result["status"] = (
        "passed"
        if code == 0
        and not result["output_rejected"]
        and result["classification"] == "unknown"
        else "failed"
    )
    return result


def version_literal(tool: str, stdout: bytes, stderr: bytes) -> str:
    require(tool in {"postgres", "initdb"})
    literal = f"{tool} (PostgreSQL) 18.6"
    require(
        stdout in {literal.encode() + b"\n", literal.encode() + b"\r\n"}
        and stderr == b""
    )
    return literal


class ReleaseRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlsplit(newurl)
        require(
            parsed.scheme == "https"
            and parsed.hostname
            in {
                "github.com",
                "release-assets.githubusercontent.com",
                "objects.githubusercontent.com",
            }
            and parsed.username is None
            and parsed.password is None
        )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_archive(private: Path) -> Path:
    artifact = ARTIFACTS["win32-x64"]["postgres"]
    require(artifact.sha256 == PIN and artifact.url == URL)
    target = private / "postgres.tar.gz"
    deadline = time.monotonic() + FETCH_SECONDS
    digest = hashlib.sha256()
    size = 0
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), ReleaseRedirect()
    )
    with opener.open(artifact.url, timeout=10) as response, target.open("xb") as output:
        require(response.status == 200)
        while chunk := response.read(1024 * 1024):
            size += len(chunk)
            require(size <= MAX_ARCHIVE and time.monotonic() < deadline)
            digest.update(chunk)
            output.write(chunk)
    require(size > 0 and digest.hexdigest() == artifact.sha256)
    return target


def archive_parts(name: str) -> tuple[str, ...]:
    require(name.isascii() and len(name) <= 240 and "\\" not in name)
    name = name.removeprefix("./")
    name = name.removesuffix("/")
    parts = tuple(name.split("/"))
    require(
        all(
            part and part not in {".", ".."} and not part.endswith((".", " "))
            for part in parts
        )
    )
    require(not re.search(r'[<>:"|?*\x00-\x1f]', name))
    require(not PurePosixPath(name).is_absolute())
    require(all(not PureWindowsPath(part).is_reserved() for part in parts))
    return parts


def extract_archive(archive: Path, destination: Path, *, deadline=None) -> None:
    require(not destination.exists() and destination.parent.is_dir())
    expanded = archive.with_suffix(".tar")
    total = 0
    deadline = (
        min(time.monotonic() + 30, deadline)
        if deadline is not None
        else time.monotonic() + 30
    )
    with gzip.open(archive, "rb") as source, expanded.open("xb") as output:
        while chunk := source.read(1024 * 1024):
            total += len(chunk)
            require(total <= MAX_EXPANDED and time.monotonic() < deadline)
            output.write(chunk)
    with tarfile.open(expanded, "r:") as bundle:
        members = []
        seen = set()
        for member in bundle:
            require(len(members) < 4096 and time.monotonic() < deadline)
            require(
                member.type in {tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE}
                and member.sparse is None
            )
            parts = archive_parts(member.name)
            key = "/".join(parts).casefold()
            require(key not in seen and 0 <= member.size <= MAX_ARCHIVE)
            seen.add(key)
            members.append((member, parts))
        require(bool(members))
        roots = {parts[0] for _, parts in members}
        strip = len(roots) == 1 and all(
            len(parts) > 1 or member.isdir() for member, parts in members
        )
        paths = [(member, parts[1:] if strip else parts) for member, parts in members]
        files = {
            "/".join(parts).casefold() for member, parts in paths if member.isfile()
        }
        for _, parts in paths:
            require(
                all(
                    "/".join(parts[:i]).casefold() not in files
                    for i in range(1, len(parts))
                )
            )
        destination.mkdir()
        for member, parts in paths:
            require(time.monotonic() < deadline)
            if not parts:
                continue
            target = destination.joinpath(*parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.extractfile(member) as source, target.open("xb") as output:
                remaining = member.size
                while remaining:
                    require(time.monotonic() < deadline)
                    chunk = source.read(min(remaining, 1024 * 1024))
                    require(bool(chunk))
                    output.write(chunk)
                    remaining -= len(chunk)
    for tool in ("postgres", "initdb"):
        with (destination / "bin" / f"{tool}.exe").open("rb") as source:
            require(source.read(2) == b"MZ")


def write_capture(private: Path, name: str, stdout: bytes, stderr: bytes) -> None:
    for suffix, raw in (("stdout", stdout), ("stderr", stderr)):
        with (private / "logs" / f"{name}.{suffix}").open("xb") as output:
            output.write(raw[:MAX_OUTPUT])


def initialize_once(
    postgres,
    bundle,
    data,
    private: Path,
    observation: dict,
    *,
    password=None,
    deadline=None,
) -> None:
    require(
        not data.postgres.exists()
        and not (data.config / "first-run.incomplete").exists()
    )
    original = postgres.run_tool

    def measured(argv, **kwargs):
        password_file = next(
            (arg[len("--pwfile=") :] for arg in argv if arg.startswith("--pwfile=")), ""
        )
        try:
            result = original(
                argv, timeout=native_timeout(INITIALIZE_SECONDS, deadline), **kwargs
            )
        except BaseException as exc:
            observation.update(empty_native(), status="error")
            try:
                if isinstance(exc, subprocess.CalledProcessError):
                    code = integer(exc.returncode)
                    observation.update(
                        status="failed",
                        child_exit_code=code,
                        windows_status=code & 0xFFFFFFFF,
                    )
                    observation.update(
                        classify_output(
                            exc.stdout or b"", exc.stderr or b"", password_file
                        )
                    )
                elif isinstance(exc, subprocess.TimeoutExpired):
                    observation["status"] = "timeout"
                elif (
                    isinstance(exc, OSError)
                    and type(getattr(exc, "winerror", None)) is int
                ):
                    observation["windows_error"] = integer(exc.winerror, 0)
                write_capture(
                    private,
                    "initialize",
                    getattr(exc, "stdout", None) or b"",
                    getattr(exc, "stderr", None) or b"",
                )
            except Exception:  # noqa: BLE001 - Preserve the initializer exception.
                observation["capture_failed"] = True
            raise
        observation.update(
            native_result(
                result.returncode, result.stdout, result.stderr, password_file
            )
        )
        try:
            write_capture(private, "initialize", result.stdout, result.stderr)
        except Exception:
            observation["capture_failed"] = True
            observation["status"] = "error"
            raise
        return result

    postgres.run_tool = measured
    try:
        postgres.initialize(
            bundle,
            data,
            password if password is not None else secrets.token_urlsafe(32),
        )
    finally:
        postgres.run_tool = original
    require(observation.get("status") == "passed" and not observation["capture_failed"])


def verify_cluster(data) -> None:
    for relative in ("PG_VERSION", "base/1/PG_VERSION"):
        with smoke.plain_path(data.postgres / relative).open("rb") as source:
            require(source.read(16) in {b"18\n", b"18\r\n"})
    for relative in ("postgresql.conf", "pg_hba.conf", "global/pg_control"):
        path = smoke.plain_path(data.postgres / relative)
        require(path.is_file() and 0 < path.stat().st_size <= MAX_OUTPUT)
    config = (data.postgres / "postgresql.conf").read_text("utf-8")
    require("\n# autogpt-desktop\nlisten_addresses = '127.0.0.1'\n" in config)
    require(
        "password_encryption = 'scram-sha-256'\n" in config  # pragma: allowlist secret
    )


def job_children(handle) -> int:
    from ctypes import wintypes

    class Accounting(ctypes.Structure):
        _fields_ = [
            ("times", ctypes.c_int64 * 4),
            ("faults", wintypes.DWORD),
            ("total", wintypes.DWORD),
            ("active", wintypes.DWORD),
            ("terminated", wintypes.DWORD),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    query = kernel.QueryInformationJobObject
    query.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.LPVOID,
    ]
    query.restype = wintypes.BOOL
    info = Accounting()
    require(
        bool(handle)
        and bool(query(handle, 1, ctypes.byref(info), ctypes.sizeof(info), None))
    )
    return integer(info.active - 1, 0, 4096)


def worker_template() -> dict:
    return {
        "status": "failed",
        "stage": "versions",
        "versions": {
            name: dict(empty_native(), literal=None) for name in ("postgres", "initdb")
        },
        "initialization": empty_native(),
        "worker_cleanup": {"active_children": None, "inspection_failed": True},
    }


def start_native(private: Path, name: str) -> None:
    pending = private / f"{name}.start-pending"
    with pending.open("x", encoding="ascii") as output:
        output.write(str(time.monotonic()))
    pending.rename(private / f"{name}.started")


def expired_native(private: Path, now: float | None = None) -> str | None:
    for name, budget in (
        ("postgres", VERSION_SECONDS),
        ("initdb", VERSION_SECONDS),
        ("initialize", INITIALIZE_SECONDS),
    ):
        path = private / f"{name}.started"
        if not path.exists() or (private / f"{name}.done").exists():
            continue
        with path.open("r", encoding="ascii") as stream:
            raw = stream.read(65)
        require(0 < len(raw) <= 64)
        started = float(raw)
        observed = time.monotonic() if now is None else now
        require(0 <= started <= observed)
        if observed - started >= budget:
            return name
    return None


def native_timeout(cap, deadline):
    if deadline is None:
        return cap
    require(type(deadline) in {int, float} and math.isfinite(deadline))
    remaining = deadline - CLEANUP_SECONDS - time.monotonic()
    if remaining <= 0:
        raise TimeoutError
    return min(cap, remaining)


def worker(private: Path, deadline=None) -> int:
    smoke.require_clean_host()
    smoke.plain_path(private)
    require(private.name == "private" and (private / "runtime/postgres").is_dir())
    (private / "worker.started").touch(exist_ok=False)
    env = smoke.private_environment(private, dict(os.environ))
    os.environ.clear()
    os.environ.update(env)
    sys.path.insert(0, str(DESKTOP / "runtime"))
    from autogpt_desktop import postgres, process
    from autogpt_desktop.layout import Bundle, DataDir

    report = worker_template()
    try:
        process.adopt_kill_on_exit_job()
        require(process._job_handle is not None)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.SetErrorMode(0x8003)
        bundle = Bundle(private / "runtime")
        data = DataDir(private / "data")
        data.prepare()
        password = None
        if deadline is not None:
            with smoke.plain_path(private / "pair.secret").open(
                "r", encoding="ascii"
            ) as source:
                password = source.read(65)
            require(bool(re.fullmatch(r"[A-Za-z0-9_-]{43}", password)))
            native_timeout(VERSION_SECONDS, deadline)
        os.chdir(bundle.root)
        for name in ("postgres", "initdb"):
            record = report["versions"][name]
            try:
                start_native(private, name)
                result = postgres.run_tool(
                    [str(bundle.postgres_bin(name)), "--version"],
                    capture_output=True,
                    env=postgres._tool_env(bundle),
                    timeout=native_timeout(VERSION_SECONDS, deadline),
                )
                record.update(
                    native_result(result.returncode, result.stdout, result.stderr)
                )
                write_capture(private, name, result.stdout, result.stderr)
                require(record["status"] == "passed")
                record["literal"] = version_literal(name, result.stdout, result.stderr)
                (private / f"{name}.done").touch(exist_ok=False)
            except subprocess.TimeoutExpired:
                record["status"] = "timeout"
                raise
            except OSError as exc:
                record["status"] = "error"
                code = getattr(exc, "winerror", None)
                if type(code) is int and 0 <= code <= 4294967295:
                    record["windows_error"] = code
                raise
            except Exception:
                if record["status"] in {"not_run", "passed"}:
                    record["status"] = "error"
                raise
        report["stage"] = "initialize"
        start_native(private, "initialize")
        initialize_once(
            postgres,
            bundle,
            data,
            private,
            report["initialization"],
            password=password,
            deadline=deadline,
        )
        (private / "initialize.done").touch(exist_ok=False)
        report["stage"] = "verify"
        verify_cluster(data)
        report.update(stage="complete", status="passed")
    except Exception:  # noqa: BLE001 - Always inspect the worker job.
        report["status"] = "failed"
    finally:
        try:
            active = job_children(process._job_handle)
            report["worker_cleanup"] = {
                "active_children": active,
                "inspection_failed": False,
            }
            if active:
                report["status"] = "failed"
        except Exception:  # noqa: BLE001 - Job inspection must fail closed.
            report["status"] = "failed"
        with (private / "worker.json").open("x", encoding="utf-8") as output:
            json.dump(report, output)
    return 0 if report["status"] == "passed" else 1


def safe_worker_result(value: dict) -> dict:
    require(type(value) is dict and set(value) == set(worker_template()))
    require(value["status"] in {"passed", "failed"} and value["stage"] in STAGES)
    require(
        type(value["versions"]) is dict
        and set(value["versions"]) == {"postgres", "initdb"}
    )
    for name, record in [
        ("initialize", value["initialization"]),
        *value["versions"].items(),
    ]:
        keys = set(empty_native()) | ({"literal"} if name != "initialize" else set())
        require(type(record) is dict and set(record) == keys)
        require(record["status"] in STATUSES and record["classification"] in CLASSES)
        for flag in ("output_rejected", "capture_failed"):
            require(type(record[flag]) is bool)
        for field in ("child_exit_code", "windows_status", "windows_error"):
            if record[field] is not None:
                integer(record[field], -2147483648 if field == "child_exit_code" else 0)
        code = record["child_exit_code"]
        require(
            record["windows_status"] == (None if code is None else code & 0xFFFFFFFF)
        )
        if name != "initialize":
            require(record["literal"] in {None, f"{name} (PostgreSQL) 18.6"})
        if record["status"] == "passed":
            require(
                code == 0
                and not record["output_rejected"]
                and not record["capture_failed"]
                and record["classification"] == "unknown"
            )
    cleanup = value["worker_cleanup"]
    require(
        type(cleanup) is dict
        and set(cleanup) == {"active_children", "inspection_failed"}
    )
    require(type(cleanup["inspection_failed"]) is bool)
    if cleanup["active_children"] is not None:
        integer(cleanup["active_children"], 0, 4096)
    if value["status"] == "passed":
        require(
            value["stage"] == "complete"
            and value["initialization"]["status"] == "passed"
        )
        require(
            all(
                record["status"] == "passed" and record["literal"] is not None
                for record in value["versions"].values()
            )
        )
        require(cleanup == {"active_children": 0, "inspection_failed": False})
    return value


def cleanup_summary(record: dict, forced=False) -> dict:
    forced = bool(
        forced
        or record["leftovers_before_cleanup"]
        or record["live_owned_during_cleanup"]
    )
    failed = bool(
        forced
        or record["surviving_owned_pids"]
        or record["remaining_listeners"]
        or record["ports_not_checked"]
        or record["observed_ports"]
        or record["inspection_failed"]
        or record["deadline_exceeded"]
    )
    return {
        "status": "failed" if failed else "passed",
        "forced_cleanup": forced,
        "survivors": len(record["surviving_owned_pids"]),
        "listeners": len(record["remaining_listeners"]),
        "inspection_failed": bool(
            record["inspection_failed"]
            or record["ports_not_checked"]
            or record["deadline_exceeded"]
        ),
    }


def supervise(private: Path, env: dict, owned, *, deadline=None) -> dict:
    report = worker_template()
    report["cleanup"] = {
        "status": "failed",
        "forced_cleanup": False,
        "survivors": None,
        "listeners": None,
        "inspection_failed": True,
    }
    report["worker_exit_code"] = None
    report["timed_out"] = False
    report["worker_report_rejected"] = False
    started = time.monotonic()
    shared_deadline = deadline
    deadline = started + NATIVE_SECONDS if deadline is None else deadline
    process = None
    forced = False
    cleanup_error = False
    supervision_failed = False
    try:
        try:
            require(
                type(deadline) in {int, float}
                and math.isfinite(deadline)
                and deadline <= started + NATIVE_SECONDS
            )
            if started >= deadline - CLEANUP_SECONDS:
                report["timed_out"] = True
                raise TimeoutError
            with (private / "logs/worker.log").open("xb") as output:
                try:
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-I",
                            "-S",
                            "-B",
                            str(Path(__file__).resolve()),
                            "--worker",
                            str(private),
                            *(
                                []
                                if shared_deadline is None
                                else ["--deadline", str(deadline)]
                            ),
                        ],
                        cwd=private / "runtime",
                        env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        shell=False,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                    for record in [
                        report["initialization"],
                        *report["versions"].values(),
                    ]:
                        record["status"] = "unknown"
                    owned.remember(owned.psutil.Process(process.pid), "launched_stage")
                    while process.poll() is None:
                        now = time.monotonic()
                        expired = expired_native(private)
                        if expired:
                            report["stage"] = (
                                "initialize" if expired == "initialize" else "versions"
                            )
                            record = (
                                report["initialization"]
                                if expired == "initialize"
                                else report["versions"][expired]
                            )
                            record["status"] = "timeout"
                        if expired or now >= min(
                            started + WORKER_SECONDS, deadline - CLEANUP_SECONDS
                        ):
                            report["timed_out"] = True
                            raise TimeoutError
                        require(
                            (private / "logs/worker.log").stat().st_size <= MAX_OUTPUT
                        )
                        owned.observe()
                        time.sleep(0.1)
                finally:
                    if process is not None:
                        try:
                            if process.poll() is None:
                                forced = True
                                process.kill()
                            process.wait(
                                timeout=max(
                                    0.01,
                                    min(3, deadline - time.monotonic()),
                                )
                            )
                        except Exception:  # noqa: BLE001 - Close the handle.
                            cleanup_error = True
                        finally:
                            try:
                                if process.returncode is not None:
                                    report["worker_exit_code"] = integer(
                                        process.returncode
                                    )
                            finally:
                                handle = getattr(process, "_handle", None)
                                if handle is not None:
                                    try:
                                        handle.Close()
                                    except Exception:  # noqa: BLE001 - Keep cleanup.
                                        cleanup_error = True
        except Exception:  # noqa: BLE001 - Still read validated worker diagnostics.
            supervision_failed = True
        try:
            try:
                report.update(
                    safe_worker_result(smoke.json_object(private / "worker.json"))
                )
            except Exception:
                report["worker_report_rejected"] = True
                raise
            require(
                not supervision_failed
                and report["worker_exit_code"] == 0
                and report["status"] == "passed"
            )
        except Exception:  # noqa: BLE001 - Always run owned-process cleanup.
            report["status"] = "failed"
    finally:
        try:
            budget = min(CLEANUP_SECONDS, max(0, deadline - time.monotonic()))
            cleanup = owned.cleanup(timeout=budget)
            forced |= bool(report["worker_cleanup"]["active_children"])
            forced |= any(
                record["status"] == "timeout"
                for record in [report["initialization"], *report["versions"].values()]
            )
            report["cleanup"] = cleanup_summary(cleanup, forced)
            if (
                supervision_failed
                or cleanup_error
                or report["worker_cleanup"]["inspection_failed"]
            ):
                report["cleanup"].update(status="failed", inspection_failed=True)
        except Exception:  # noqa: BLE001 - Preserve native state on cleanup errors.
            report["cleanup"].update(
                status="failed", inspection_failed=True, forced_cleanup=forced
            )
        report["native_elapsed_ms"] = min(
            3600000, max(0, int((time.monotonic() - started) * 1000))
        )
        if (
            report["cleanup"]["status"] != "passed"
            or report["native_elapsed_ms"] >= (deadline - started) * 1000
        ):
            if report["status"] == "passed":
                report["stage"] = "cleanup"
            report["status"] = "failed"
    return report


def prepare_legacy_work_dir(work: Path, artifact: Path) -> Path:
    """Diagnostic-only Python 0700 fixture; never used by product acceptance."""
    smoke.check_new_work_dir(work, artifact)
    work.mkdir(mode=0o700)
    private = work / "private"
    private.mkdir(mode=0o700)
    for name in ("logs", "home", "appdata", "localappdata", "temp", "data"):
        (private / name).mkdir(mode=0o700)
    return private


def control_sibling(work: Path) -> Path:
    smoke.plain_path(work)
    require(work.is_dir() and bool(work.name))
    suffix = "c" if work.name.endswith("b") else "b"
    control = work.with_name(work.name[:-1] + suffix)
    smoke.plain_path(control)
    require(control != work and not control.exists())
    require(control.parent == work.parent and len(str(control)) == len(str(work)))
    for protected in (BUILD, DESKTOP.parents[1]):
        require(
            not control.is_relative_to(protected)
            and not protected.is_relative_to(control)
        )
    return control


def payload_manifest(root: Path, deadline) -> dict:
    smoke.plain_path(root)
    require(root.is_dir())
    result = {}
    total = 0
    for path in root.rglob("*"):
        acl.check_time(deadline - CLEANUP_SECONDS)
        smoke.plain_path(path)
        require(len(result) < 8192)
        if path.is_dir():
            value = ("directory", None)
        else:
            require(path.is_file())
            size = path.stat().st_size
            total += size
            require(size <= MAX_ARCHIVE and total <= MAX_EXPANDED)
            value = ("file", smoke.sha256(path))
        result[path.relative_to(root).as_posix()] = value
    return result


def prepare_control(private, archive, env, payload, deadline):
    work = private.parent
    control = control_sibling(work)
    security = acl.PairACL(work, control, deadline - CLEANUP_SECONDS)
    security.inspect_baseline(payload)
    security.create_private_tree()
    other = control / "private"
    with (private / "pair.secret").open("rb") as source, (other / "pair.secret").open(
        "xb"
    ) as target:
        password = source.read(65)
        require(bool(re.fullmatch(rb"[A-Za-z0-9_-]{43}", password)))
        target.write(password)
    acl.check_time(deadline - CLEANUP_SECONDS)
    require(archive == private / "postgres.tar.gz" and smoke.sha256(archive) == PIN)
    control_archive = other / "postgres.tar.gz"
    with archive.open("rb") as source, control_archive.open("xb") as target:
        size = 0
        while chunk := source.read(1024 * 1024):
            acl.check_time(deadline - CLEANUP_SECONDS)
            size += len(chunk)
            require(size <= MAX_ARCHIVE)
            target.write(chunk)
    require(size > 0 and smoke.sha256(control_archive) == PIN)
    (other / "runtime").mkdir()
    extract_archive(
        control_archive, other / "runtime/postgres", deadline=deadline - CLEANUP_SECONDS
    )
    require(payload_manifest(private / "runtime/postgres", deadline) == payload)
    require(payload_manifest(other / "runtime/postgres", deadline) == payload)
    sys.path.insert(0, str(DESKTOP / "runtime"))
    from autogpt_desktop.layout import DataDir

    DataDir(other / "data").prepare()
    other_env = smoke.private_environment(other, dict(os.environ))
    other_env.update(GITHUB_ACTIONS="true", RUNNER_ENVIRONMENT="github-hosted")
    require(
        {
            key: value.replace(str(other), str(private)).replace(
                str(other)[len(other.drive) :], str(private)[len(private.drive) :]
            )
            for key, value in other_env.items()
        }
        == env
    )
    evidence = security.compare(payload)
    acl.check_time(deadline - CLEANUP_SECONDS)
    return other, other_env, evidence


def clean_arm(report):
    return (
        report["cleanup"]
        == {
            "status": "passed",
            "forced_cleanup": False,
            "survivors": 0,
            "listeners": 0,
            "inspection_failed": False,
        }
        and report["worker_cleanup"]
        == {"active_children": 0, "inspection_failed": False}
        and not report["timed_out"]
        and not report["worker_report_rejected"]
        and report["worker_exit_code"] in {0, 1}
        and all(
            record["status"] == "passed"
            and record["literal"] == f"{name} (PostgreSQL) 18.6"
            for name, record in report["versions"].items()
        )
        and not report["initialization"]["output_rejected"]
        and not report["initialization"]["capture_failed"]
    )


def comparison_template():
    return {
        "control_status": "not_run",
        "reason": "baseline_not_reproduced",
        "outcome": "inconclusive",
        "inputs_match": False,
        "security": None,
        "control_result": None,
        "sequence_elapsed_ms": 0,
        "timing_scope": "pair_including_preparation_and_cleanup",
    }


def run_pair(private, archive, env):
    started = time.monotonic()
    deadline = started + NATIVE_SECONDS
    comparison = comparison_template()
    with (private / "pair.secret").open("x", encoding="ascii") as output:
        output.write(secrets.token_urlsafe(32))
    payload = payload_manifest(private / "runtime/postgres", deadline)
    owned = smoke.OwnedProcesses(private / "runtime", private / "data")
    baseline = supervise(private, env, owned, deadline=deadline)
    baseline["acl_comparison"] = comparison
    try:
        if not clean_arm(baseline):
            comparison["reason"] = "baseline_guard_failed"
            return baseline
        if not (
            baseline["status"] == "failed"
            and baseline["stage"] == "initialize"
            and baseline["initialization"]["status"] == "failed"
            and baseline["initialization"]["windows_status"] == 0xC0000135
        ):
            return baseline
        comparison["reason"] = "guard_failed"
        acl.check_time(deadline - CLEANUP_SECONDS)
        other, other_env, evidence = prepare_control(
            private, archive, env, payload, deadline
        )
        comparison["security"] = evidence
        acl.check_time(deadline - CLEANUP_SECONDS)
        owned = smoke.OwnedProcesses(other / "runtime", other / "data")
        result = supervise(other, other_env, owned, deadline=deadline)
        comparison["control_result"] = result
        comparison["control_status"] = result["status"]
        require(payload_manifest(other / "runtime/postgres", deadline) == payload)
        require(payload_manifest(private / "runtime/postgres", deadline) == payload)
        comparison["inputs_match"] = True
        comparison["reason"] = "completed"
        if clean_arm(result):
            if result["status"] == "passed":
                comparison["outcome"] = "acl_dependent"
            elif result["initialization"]["windows_status"] == 0xC0000135:
                comparison["outcome"] = "same_status"
    except TimeoutError:
        comparison.update(reason="deadline", outcome="inconclusive")
    except Exception:  # noqa: BLE001 - Retain the authoritative baseline failure.
        comparison.update(reason="guard_failed", outcome="inconclusive")
    finally:
        elapsed = max(0, int((time.monotonic() - started) * 1000))
        comparison["sequence_elapsed_ms"] = min(3600000, elapsed)
        if elapsed >= NATIVE_SECONDS * 1000:
            baseline["status"] = "failed"
            comparison.update(reason="deadline", outcome="inconclusive")
    return baseline


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--work-dir", type=Path)
    group.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--deadline", type=float, help=argparse.SUPPRESS)
    parser.add_argument(
        "--legacy-acl-comparison",
        action="store_true",
        help="diagnostic only: compare legacy Python 0700 roots; always exits nonzero",
    )
    args = parser.parse_args(argv)
    if args.worker and args.legacy_acl_comparison:
        parser.error("--legacy-acl-comparison requires --work-dir")
    if args.worker:
        try:
            return worker(args.worker, args.deadline)
        except Exception:  # noqa: BLE001 - Do not expose private worker exceptions.
            return 1
    work = args.work_dir
    private = None
    summary = {
        "schema_version": 1,
        "status": "failed",
        "stage": "host",
        "source_commit": None,
        "source_sha256": {},
        "expected_archive_sha256": PIN,
        "archive_sha256": None,
        "acl_comparison": comparison_template(),
    }
    try:
        require(args.deadline is None)
        smoke.require_clean_host()
        prepare = (
            prepare_legacy_work_dir
            if args.legacy_acl_comparison
            else smoke.prepare_work_dir
        )
        private = prepare(work, BUILD)
        commit = os.environ.get("GITHUB_SHA", "")
        require(bool(re.fullmatch("[0-9a-f]{40}", commit)))
        summary["source_commit"] = commit
        summary["source_sha256"] = {
            name: smoke.sha256(DESKTOP / path) for name, path in SOURCES.items()
        }
        env = smoke.private_environment(private, dict(os.environ))
        env.update(GITHUB_ACTIONS="true", RUNNER_ENVIRONMENT="github-hosted")
        summary["stage"] = "artifact"
        archive = fetch_archive(private)
        summary["archive_sha256"] = smoke.sha256(archive)
        (private / "runtime").mkdir()
        extract_archive(archive, private / "runtime/postgres")
        summary["stage"] = "versions"
        if args.legacy_acl_comparison:
            summary.update(run_pair(private, archive, env))
            summary["status"] = "failed"
        else:
            owned = smoke.OwnedProcesses(private / "runtime", private / "data")
            summary.update(supervise(private, env, owned))
    except Exception:  # noqa: BLE001 - Record failure without private exception text.
        summary["status"] = "failed"
    if private is not None:
        try:
            pending = private / "result.pending"
            with pending.open("x", encoding="utf-8") as output:
                json.dump(summary, output, indent=2)
                output.write("\n")
            require(not (work / "result.json").exists())
            pending.rename(work / "result.json")
        except Exception:  # noqa: BLE001 - Summary I/O failure must remain a failure.
            return 1
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
