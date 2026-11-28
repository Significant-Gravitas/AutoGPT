"""Bounded, explicit Windows host-filesystem acceptance on Python 3.13."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET
from contextlib import ExitStack
from pathlib import Path

BUILD = Path(__file__).resolve().parent
BACKEND = BUILD.parent.parent / "backend"
TESTS = (
    "backend/copilot/host_fs_portable_test.py",
    "backend/copilot/host_fs_windows_native_checks.py",
)
CALLER_TEST = "desktop/build/backend_host_fs_windows_callers.py"
CALLER_SOURCES = (
    "backend/copilot/sdk/cli_prelaunch_turn_test.py",
    "backend/copilot/sdk/retry_scenarios_test.py",
    "backend/copilot/sdk/conftest.py",
    "backend/copilot/sdk/transcript_builder.py",
    "backend/copilot/tools/sandbox.py",
    "backend/copilot/tools/workspace_files.py",
    "backend/copilot/transcript.py",
)
SOURCES = (
    "backend/copilot/host_fs.py",
    "backend/copilot/host_fs_windows.py",
    "backend/copilot/windows_handles.py",
    "backend/copilot/host_private_directory.py",
    "backend/copilot/windows_private.py",
    "backend/copilot/sdk/service.py",
    "backend/copilot/tools/workdir.py",
    *TESTS,
)
TEST_SECONDS = 180
CLEANUP_SECONDS = 30
MAX_ENTRIES = 4096
MAX_LOG_BYTES = 2 * 1024 * 1024


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def require_native() -> None:
    require(
        os.name == "nt" and sys.version_info[:2] == (3, 13),
        "windows_python313_required",
    )


def selected_tests(callers=False) -> dict[str, Path]:
    paths = {name: BACKEND / name for name in TESTS}
    if callers:
        paths[CALLER_TEST] = BUILD / "backend_host_fs_windows_callers.py"
    return paths


def source_hashes(callers=False) -> dict[str, str]:
    paths = {name: BACKEND / name for name in SOURCES}
    paths["desktop/build/host_fs_windows_check.py"] = Path(__file__)
    if callers:
        paths.update({name: BACKEND / name for name in CALLER_SOURCES})
        paths[CALLER_TEST] = BUILD / "backend_host_fs_windows_callers.py"
        for name in ("layout.py", "settings.py"):
            paths[f"desktop/runtime/autogpt_desktop/{name}"] = (
                BUILD.parent / "runtime/autogpt_desktop" / name
            )
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
    }


class WindowsJob:
    def __init__(self):
        from ctypes import wintypes as w

        class Limits(ctypes.Structure):
            _fields_ = [
                ("times", ctypes.c_int64 * 2),
                ("flags", w.DWORD),
                ("working_sets", ctypes.c_size_t * 2),
                ("active", w.DWORD),
                ("affinity", ctypes.c_size_t),
                ("priority", w.DWORD),
                ("scheduling", w.DWORD),
            ]

        class Extended(ctypes.Structure):
            _fields_ = [
                ("basic", Limits),
                ("io", ctypes.c_uint64 * 6),
                ("memory", ctypes.c_size_t * 4),
            ]

        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        signatures = (
            ("CreateJobObjectW", w.HANDLE, (w.LPVOID, w.LPCWSTR)),
            (
                "SetInformationJobObject",
                w.BOOL,
                (w.HANDLE, ctypes.c_int, w.LPVOID, w.DWORD),
            ),
            ("AssignProcessToJobObject", w.BOOL, (w.HANDLE, w.HANDLE)),
            ("TerminateJobObject", w.BOOL, (w.HANDLE, w.UINT)),
            (
                "QueryInformationJobObject",
                w.BOOL,
                (w.HANDLE, ctypes.c_int, w.LPVOID, w.DWORD, w.LPVOID),
            ),
            ("CloseHandle", w.BOOL, (w.HANDLE,)),
        )
        for name, restype, argtypes in signatures:
            function = getattr(self.kernel, name)
            function.restype, function.argtypes = restype, argtypes
        self.handle = self.kernel.CreateJobObjectW(None, None)
        require(bool(self.handle), "job_create_failed")
        limits = Extended()
        limits.basic.flags = 0x2000
        try:
            require(
                bool(
                    self.kernel.SetInformationJobObject(
                        self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
                    )
                ),
                "job_limits_failed",
            )
        except BaseException:
            self.close()
            raise

    def assign(self, process) -> None:
        require(
            bool(
                self.kernel.AssignProcessToJobObject(self.handle, int(process._handle))
            ),
            "job_assignment_failed",
        )

    def active(self) -> int:
        info = ctypes.create_string_buffer(48)
        require(
            bool(
                self.kernel.QueryInformationJobObject(
                    self.handle, 1, info, len(info), None
                )
            ),
            "job_query_failed",
        )
        return struct.unpack_from("<I", info.raw, 40)[0]

    def terminate(self) -> None:
        require(
            bool(self.kernel.TerminateJobObject(self.handle, 1)),
            "job_termination_failed",
        )

    def close(self) -> None:
        if self.handle:
            require(bool(self.kernel.CloseHandle(self.handle)), "job_close_failed")
            self.handle = None


def directory_names(wh, handle: int):
    buffer = ctypes.create_string_buffer(65536)
    restart = True
    while True:
        if not wh.get_info(handle, 11 if restart else 10, buffer, len(buffer)):
            if ctypes.get_last_error() == 18:
                return
            raise ctypes.WinError(ctypes.get_last_error())
        restart = False
        offset = 0
        while True:
            following, length = struct.unpack_from("<I56xI", buffer.raw, offset)
            require(
                length % 2 == 0
                and 0 < length <= 510
                and offset + 104 + length <= len(buffer),
                "invalid_directory_record",
            )
            name = buffer.raw[offset + 104 : offset + 104 + length].decode("utf-16-le")
            if name not in (".", ".."):
                require(
                    not any(char in name for char in "\\/\0:"), "invalid_directory_name"
                )
                yield name
            if following == 0:
                break
            require(
                following >= 104 + length and offset + following < len(buffer),
                "invalid_directory_offset",
            )
            offset += following


def retry_sharing(operation, deadline: float):
    while True:
        try:
            return operation()
        except OSError as exc:
            if (
                getattr(exc, "winerror", None) not in (32, 33)
                or time.monotonic() >= deadline
            ):
                raise
            time.sleep(min(0.05, max(0, deadline - time.monotonic())))


def remove_contents(
    wh, handle: int, deadline: float, budget: list[int], depth=0
) -> None:
    require(depth <= 20, "cleanup_depth_exceeded")
    for name in directory_names(wh, handle):
        require(
            time.monotonic() < deadline and budget[0] > 0, "cleanup_budget_exceeded"
        )
        budget[0] -= 1
        child = retry_sharing(
            lambda name=name: wh.relative(
                handle,
                name,
                wh.DELETE | wh.READ_ATTRIBUTES | 1,
                wh.FILE_OPEN,
                allow_link=True,
            ),
            deadline,
        )
        try:
            attrs = wh.attributes(child)
            if attrs & wh.DIRECTORY and not attrs & wh.REPARSE:
                remove_contents(wh, child, deadline, budget, depth + 1)
            delete = ctypes.c_uint32(1 | 0x10)
            require(
                bool(
                    wh.set_info(child, 21, ctypes.byref(delete), ctypes.sizeof(delete))
                ),
                "cleanup_delete_failed",
            )
        finally:
            wh.close(child)


class PrivateFixture:
    def __init__(self, drive: str):
        require(bool(re.fullmatch(r"[A-Za-z]:", drive)), "local_volume_required")
        sys.path.insert(0, str(BACKEND))
        from backend.copilot import windows_handles as wh
        from backend.copilot import windows_private as private

        self.wh = wh
        self.stack = ExitStack()
        self.handle = None
        self.path = Path(drive + "\\") / ("autogpt-host-fs-" + uuid.uuid4().hex)
        try:
            self.parent = wh.open_root(drive, security=True)
            self.stack.callback(wh.close, self.parent)
            user = private.current_user_sid()
            private.verify_acl(self.parent, user, private=False)
            with private.security_descriptor(user) as descriptor:
                self.handle = wh.relative(
                    self.parent,
                    self.path.name,
                    wh.DIRECTORY_ACCESS | wh.READ_CONTROL | 1,
                    wh.FILE_CREATE,
                    directory=True,
                    descriptor=descriptor,
                )
            wh.inspect(self.handle, str(self.path), directory=True)
            private.verify_acl(self.handle, user, private=True)
            self.identity = self.file_identity(self.handle)
        except BaseException:
            # A failed exclusive create never authorizes removal of a collision.
            if self.handle is not None:
                wh.close(self.handle)
                print(
                    f"Owned fixture retained after validation failure: {self.path}",
                    file=sys.stderr,
                )
            self.stack.close()
            raise

    def file_identity(self, handle: int) -> bytes:
        identity = ctypes.create_string_buffer(24)
        require(
            bool(self.wh.get_info(handle, 18, identity, len(identity))),
            "fixture_identity_failed",
        )
        return identity.raw

    def cleanup(self) -> None:
        wh = self.wh
        try:
            require(
                self.file_identity(self.handle) == self.identity,
                "fixture_identity_changed",
            )
            remove_contents(
                wh, self.handle, time.monotonic() + CLEANUP_SECONDS, [MAX_ENTRIES]
            )
            wh.close(self.handle)
            self.handle = None
            handle = wh.relative(
                self.parent,
                self.path.name,
                wh.DIRECTORY_ACCESS | wh.DELETE,
                wh.FILE_OPEN,
                directory=True,
            )
            try:
                wh.inspect(handle, str(self.path), directory=True)
                require(
                    self.file_identity(handle) == self.identity,
                    "fixture_identity_changed",
                )
                delete = ctypes.c_ubyte(1)
                require(
                    bool(
                        wh.set_info(
                            handle, 4, ctypes.byref(delete), ctypes.sizeof(delete)
                        )
                    ),
                    "fixture_delete_failed",
                )
            finally:
                wh.close(handle)
        finally:
            if self.handle is not None:
                wh.close(self.handle)
                self.handle = None
            self.stack.close()


class TestInventory:
    def __init__(self, callers=False):
        self.paths = {
            path.resolve(): name for name, path in selected_tests(callers).items()
        }
        self.collected = {name: 0 for name in selected_tests(callers)}
        self.passed = 0
        self.failed = 0
        self.skipped = 0

    def pytest_collection_modifyitems(self, items):
        for item in items:
            path = Path(item.path).resolve()
            require(path in self.paths, "unexpected_test_module")
            self.collected[self.paths[path]] += 1

    def pytest_collectreport(self, report):
        self.failed += int(report.failed)
        self.skipped += int(report.skipped)

    def pytest_runtest_logreport(self, report):
        self.failed += int(report.failed)
        self.skipped += int(report.skipped)
        self.passed += int(report.when == "call" and report.passed)


def worker(root: Path, *, callers=False) -> int:
    require(sys.stdin.readline() == "run\n", "worker_not_adopted")
    require_native()
    os.environ.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTHON_DOTENV_DISABLED="1")
    if callers:
        sys.path.insert(0, str(BUILD.parent / "runtime"))
        from autogpt_desktop.layout import Bundle
        from autogpt_desktop.settings import _bundled_tools

        bundle = Bundle(Path(sys.executable).resolve().parent.parent)
        require(bundle.ffmpeg.is_file(), "bundled_ffmpeg_missing")
        os.environ.update(_bundled_tools(bundle))
    sys.path.insert(0, str(BACKEND))
    import pytest

    inventory = TestInventory(callers)
    before = source_hashes(callers)
    code = pytest.main(
        [
            "-q",
            "--noconftest",
            "-p",
            "no:cacheprovider",
            "-c",
            str(BUILD.parent / "runtime/pyproject.toml"),
            "-o",
            "addopts=",
            "--basetemp",
            str(root / "pytest"),
            *(["-p", "pytest_asyncio.plugin"] if callers else []),
            *(str(path) for path in selected_tests(callers).values()),
        ],
        plugins=[inventory],
    )
    result = {
        "exit_code": int(code),
        "collected": inventory.collected,
        "passed": inventory.passed,
        "failed": inventory.failed,
        "skipped": inventory.skipped,
        "sources": before,
        "sources_unchanged": before == source_hashes(callers),
    }
    (root / "worker.json").write_text(json.dumps(result), encoding="utf-8")
    return int(code)


def supervise(
    root: Path,
    *,
    callers=False,
    coverage=False,
    timeout=TEST_SECONDS,
    job_factory=WindowsJob,
    popen=subprocess.Popen,
) -> dict:
    job = job_factory()
    process = None
    result = {
        "exit_code": None,
        "timed_out": False,
        "forced_cleanup": False,
        "active_processes": None,
    }
    try:
        with (root / "pytest.log").open("xb") as output:
            process = popen(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    *(
                        [
                            "-m",
                            "coverage",
                            "run",
                            "--branch",
                            "--source",
                            str(BACKEND / "backend"),
                            "--data-file",
                            str(root / ".coverage"),
                        ]
                        if coverage
                        else []
                    ),
                    str(Path(__file__).resolve()),
                    "--worker",
                    str(root),
                    *(["--callers"] if callers else []),
                ],
                stdin=subprocess.PIPE,
                stdout=output,
                stderr=subprocess.STDOUT,
                cwd=BACKEND,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            job.assign(process)
            process.stdin.write(b"run\n")
            process.stdin.close()
            deadline = time.monotonic() + timeout
            while process.poll() is None:
                require(
                    (root / "pytest.log").stat().st_size <= MAX_LOG_BYTES,
                    "worker_log_limit",
                )
                if time.monotonic() >= deadline:
                    result["timed_out"] = True
                    break
                time.sleep(0.05)
    finally:
        try:
            if process is not None:
                if process.poll() is None or job.active():
                    result["forced_cleanup"] = True
                    try:
                        job.terminate()
                    finally:
                        # Also covers assignment failure before the handshake.
                        if process.poll() is None:
                            process.kill()
                process.wait(timeout=5)
                result["exit_code"] = process.returncode
                deadline = time.monotonic() + 5
                while job.active() and time.monotonic() < deadline:
                    time.sleep(0.05)
                result["active_processes"] = job.active()
                require(result["active_processes"] == 0, "owned_processes_survived")
        finally:
            job.close()
            if process is not None:
                if process.stdin is not None and not process.stdin.closed:
                    process.stdin.close()
                handle = getattr(process, "_handle", None)
                if handle is not None:
                    handle.Close()
    return result


def export_coverage(root: Path, destination: Path) -> None:
    import coverage

    backend_source = BACKEND / "backend"
    measured = coverage.Coverage(
        data_file=str(root / ".coverage"),
        source=[str(backend_source)],
        config_file=False,
    )
    measured.load()
    require(bool(measured.get_data().measured_files()), "native_coverage_missing")
    destination.parent.mkdir(parents=True, exist_ok=True)
    measured.xml_report(outfile=str(destination))
    report = ET.parse(destination)
    for item in report.findall(".//class"):
        file = (backend_source / item.attrib["filename"]).resolve()
        require(file.is_relative_to(backend_source), "unexpected_coverage_source")
        item.set("filename", file.relative_to(BACKEND.parent.parent).as_posix())
    for source in report.findall("sources/source"):
        source.text = "."
    report.write(destination, encoding="utf-8", xml_declaration=True)


def accepted(report: dict, sources: dict, *, callers=False) -> bool:
    counts = report.get("collected", {})
    return (
        set(counts) == set(selected_tests(callers))
        and all(type(count) is int and count > 0 for count in counts.values())
        and report.get("passed") == sum(counts.values())
        and report.get("failed") == 0
        and report.get("skipped") == 0
        and report.get("exit_code") == 0
        and report.get("sources") == sources
        and report.get("sources_unchanged") is True
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path)
    parser.add_argument(
        "--coverage", type=Path, help="Export native test coverage as XML"
    )
    parser.add_argument("--fixture-drive", default=Path(sys.executable).drive)
    parser.add_argument("--worker", type=Path)
    parser.add_argument(
        "--callers",
        action="store_true",
        help="Also test imported checkout callers using the bundled backend dependencies",
    )
    args = parser.parse_args(argv)
    if args.worker is not None:
        return worker(args.worker, callers=args.callers)
    require(args.result is not None, "result_required")
    result = {
        "status": "failed",
        "cleanup": "not_created",
        "python": sys.version,
        "executable_sha256": hashlib.sha256(
            Path(sys.executable).read_bytes()
        ).hexdigest(),
        "github_sha": os.environ.get("GITHUB_SHA"),
        "sources": source_hashes(args.callers),
        "scope": (
            "checkout_callers_with_native_primitives"
            if args.callers
            else "native_primitives"
        ),
    }
    fixture = None
    cleanup_safe = True
    try:
        require_native()
        fixture = PrivateFixture(args.fixture_drive)
        result["cleanup"] = "pending"
        cleanup_safe = False
        result["worker"] = supervise(
            fixture.path, callers=args.callers, coverage=args.coverage is not None
        )
        cleanup_safe = result["worker"]["active_processes"] == 0
        with (fixture.path / "worker.json").open("rb") as worker_report:
            raw = worker_report.read(65537)
        require(len(raw) <= 65536, "worker_report_limit")
        report = json.loads(raw)
        result["tests"] = report
        require(
            accepted(report, result["sources"], callers=args.callers),
            "native_tests_failed_or_skipped",
        )
        require(
            result["worker"]
            == {
                "exit_code": 0,
                "timed_out": False,
                "forced_cleanup": False,
                "active_processes": 0,
            },
            "worker_did_not_finish_cleanly",
        )
        require(result["sources"] == source_hashes(args.callers), "sources_changed")
        if args.coverage is not None:
            export_coverage(fixture.path, args.coverage)
        result["status"] = "passed"
    except Exception as exc:  # noqa: BLE001 - Preserve failed acceptance evidence.
        result["error"] = type(exc).__name__
        if fixture is not None:
            try:
                with (fixture.path / "pytest.log").open("rb") as output:
                    output.seek(max(0, output.seek(0, 2) - 16384))
                    print(
                        output.read(16384).decode("utf-8", errors="replace"),
                        file=sys.stderr,
                    )
            except OSError:
                pass
        print(
            f"Windows host-filesystem gate failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
    finally:
        if fixture is not None:
            try:
                require(cleanup_safe, "process_cleanup_unproven_fixture_retained")
                fixture.cleanup()
                result["cleanup"] = "passed"
            except Exception as exc:  # noqa: BLE001 - Fail closed.
                result.update(
                    status="failed", cleanup="failed", cleanup_error=type(exc).__name__
                )
                print(
                    f"Owned fixture retained after cleanup failure: {fixture.path}: {exc}",
                    file=sys.stderr,
                )
        args.result.parent.mkdir(parents=True, exist_ok=True)
        with args.result.open("x", encoding="utf-8") as output:
            json.dump(result, output, indent=2)
        print(json.dumps({key: result[key] for key in ("status", "cleanup")}))
    return int(result["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
