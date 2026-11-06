"""Gate failures stay failures; cleanup never traverses a reparse point."""

import ctypes
import importlib.util
import io
import json
import os
import struct
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

SPEC = importlib.util.spec_from_file_location(
    "host_fs_windows_check", Path(__file__).parents[1] / "host_fs_windows_check.py"
)
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def passing_report():
    return {
        "collected": {name: 2 for name in gate.TESTS},
        "passed": 4,
        "failed": 0,
        "skipped": 0,
        "exit_code": 0,
        "sources": {"fixture": "a" * 64},
        "sources_unchanged": True,
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"exit_code": 1},
        {"failed": 1},
        {"skipped": 1},
        {"passed": 3},
        {"collected": {}},
        {"collected": {gate.TESTS[0]: 4}},
        {"collected": {name: 0 for name in gate.TESTS}},
        {"sources_unchanged": False},
        {"sources": {}},
    ],
)
def test_incomplete_native_evidence_is_rejected(changes):
    report = passing_report()
    sources = report["sources"]
    assert gate.accepted(report, sources)
    report.update(changes)
    assert not gate.accepted(report, sources)


def test_collection_and_runtime_skips_are_observed():
    inventory = gate.TestInventory()
    inventory.pytest_collectreport(SimpleNamespace(failed=False, skipped=True))
    inventory.pytest_runtest_logreport(
        SimpleNamespace(failed=False, skipped=True, when="setup", passed=False)
    )
    assert inventory.skipped == 2
    assert inventory.passed == 0


def test_callers_cannot_be_omitted_from_acceptance():
    report = passing_report()
    assert not gate.accepted(report, report["sources"], callers=True)
    report["collected"][gate.CALLER_TEST] = 7
    report["passed"] += 7
    assert gate.accepted(report, report["sources"], callers=True)


def test_caller_inventory_uses_explicit_checkout_file():
    inventory = gate.TestInventory(callers=True)
    path = gate.BUILD / "backend_host_fs_windows_callers.py"
    inventory.pytest_collection_modifyitems([SimpleNamespace(path=path)])
    assert inventory.collected[gate.CALLER_TEST] == 1
    assert all(inventory.collected[name] == 0 for name in gate.TESTS)


def test_caller_worker_configures_relocated_ffmpeg_before_collection(
    tmp_path, monkeypatch
):
    from autogpt_desktop.layout import Bundle

    bundle = Bundle((tmp_path / "runtime").resolve())
    bundle.tools_bin.mkdir(parents=True)
    bundle.ffmpeg.write_bytes(b"synthetic tool; not executed")
    monkeypatch.setattr(gate.sys, "executable", str(bundle.root / "python/python.exe"))
    monkeypatch.setattr(gate.sys, "stdin", io.StringIO("run\n"))
    monkeypatch.setattr(gate, "require_native", lambda: None)
    monkeypatch.setattr(gate.sys, "path", list(gate.sys.path))
    monkeypatch.setenv("PATH", "original-tools")
    monkeypatch.setenv("IMAGEIO_FFMPEG_EXE", "stale-host-ffmpeg")
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")

    def collect(args, plugins):
        assert os.environ["IMAGEIO_FFMPEG_EXE"] == str(bundle.ffmpeg)
        assert os.environ["PATH"] == os.pathsep.join(
            [str(bundle.tools_bin), "original-tools"]
        )
        assert "pytest_asyncio.plugin" in args
        assert str(gate.BUILD / "backend_host_fs_windows_callers.py") in args
        return 0

    monkeypatch.setattr(pytest, "main", collect)
    assert gate.worker(tmp_path, callers=True) == 0


def test_caller_worker_rejects_missing_bundled_ffmpeg(tmp_path, monkeypatch):
    monkeypatch.setattr(
        gate.sys, "executable", str(tmp_path / "runtime/python/python.exe")
    )
    monkeypatch.setattr(gate.sys, "stdin", io.StringIO("run\n"))
    monkeypatch.setattr(gate, "require_native", lambda: None)
    monkeypatch.setattr(gate.sys, "path", list(gate.sys.path))
    monkeypatch.setenv("IMAGEIO_FFMPEG_EXE", "host-tool-must-not-cover-missing-bundle")
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    collect = Mock(return_value=0)
    monkeypatch.setattr(pytest, "main", collect)
    with pytest.raises(RuntimeError, match="bundled_ffmpeg_missing"):
        gate.worker(tmp_path, callers=True)
    collect.assert_not_called()


def test_component_worker_does_not_require_or_replace_bundled_tools(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(gate.sys, "executable", str(tmp_path / "python.exe"))
    monkeypatch.setattr(gate.sys, "stdin", io.StringIO("run\n"))
    monkeypatch.setattr(gate, "require_native", lambda: None)
    monkeypatch.setattr(gate.sys, "path", list(gate.sys.path))
    monkeypatch.setenv("IMAGEIO_FFMPEG_EXE", "untouched")
    monkeypatch.setenv("PATH", "original-tools")
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setattr(pytest, "main", Mock(return_value=0))
    assert gate.worker(tmp_path) == 0
    assert os.environ["IMAGEIO_FFMPEG_EXE"] == "untouched"
    assert os.environ["PATH"] == "original-tools"


class FakeProcess:
    def __init__(self, code):
        self.returncode = code
        self.stdin = Mock(closed=False)
        self._handle = Mock()
        self.killed = False

    def poll(self):
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = 1

    def wait(self, timeout):
        assert timeout <= 5
        assert self.returncode is not None


class FakeJob:
    def __init__(self, active=0):
        self.count = active
        self.closed = False
        self.terminated = False

    def assign(self, process):
        assert not process.stdin.write.called

    def active(self):
        return self.count

    def terminate(self):
        self.terminated = True
        self.count = 0

    def close(self):
        self.closed = True


@pytest.mark.parametrize("code", [0, 1])
def test_worker_exit_code_and_handles_preserved(tmp_path, code):
    process, job = FakeProcess(code), FakeJob()
    report = gate.supervise(
        tmp_path, popen=Mock(return_value=process), job_factory=lambda: job
    )
    assert report["exit_code"] == code
    assert report["active_processes"] == 0
    assert not report["forced_cleanup"]
    process.stdin.write.assert_called_once_with(b"run\n")
    process._handle.Close.assert_called_once()
    assert job.closed


def test_caller_mode_reaches_worker(tmp_path):
    popen = Mock(return_value=FakeProcess(0))
    gate.supervise(tmp_path, callers=True, popen=popen, job_factory=FakeJob)
    assert popen.call_args.args[0][-1] == "--callers"


def test_timeout_terminates_only_owned_job_and_worker(tmp_path):
    process, job = FakeProcess(None), FakeJob(1)
    report = gate.supervise(
        tmp_path, timeout=0, popen=Mock(return_value=process), job_factory=lambda: job
    )
    assert report["timed_out"] and report["forced_cleanup"]
    assert job.terminated and process.killed and job.closed
    assert report["active_processes"] == 0


def test_leaked_child_fails_even_when_pytest_exits_zero(tmp_path):
    process, job = FakeProcess(0), FakeJob(1)
    report = gate.supervise(
        tmp_path, popen=Mock(return_value=process), job_factory=lambda: job
    )
    assert report["exit_code"] == 0
    assert report["forced_cleanup"] and job.terminated


def test_assignment_failure_never_releases_worker(tmp_path):
    process, job = FakeProcess(None), FakeJob()
    job.assign = Mock(side_effect=OSError("assignment failed"))
    with pytest.raises(OSError):
        gate.supervise(
            tmp_path, popen=Mock(return_value=process), job_factory=lambda: job
        )
    process.stdin.write.assert_not_called()
    assert process.killed and job.closed


def test_log_bound_terminates_owned_worker(tmp_path):
    process, job = FakeProcess(None), FakeJob(1)
    with patch.object(gate, "MAX_LOG_BYTES", -1), pytest.raises(
        RuntimeError, match="worker_log_limit"
    ):
        gate.supervise(
            tmp_path, popen=Mock(return_value=process), job_factory=lambda: job
        )
    assert process.killed and job.terminated and job.closed


def fake_handles(attrs):
    return SimpleNamespace(
        DELETE=0x10000,
        READ_ATTRIBUTES=0x80,
        DIRECTORY=0x10,
        REPARSE=0x400,
        FILE_OPEN=1,
        relative=Mock(return_value=2),
        attributes=Mock(return_value=attrs),
        set_info=Mock(return_value=True),
        close=Mock(),
    )


@pytest.mark.parametrize("attrs", [0, 0x400, 0x410])
def test_cleanup_unlinks_leaf_or_reparse_without_traversal(attrs):
    wh = fake_handles(attrs)
    with patch.object(gate, "directory_names", return_value=iter(["child"])) as names:
        gate.remove_contents(wh, 1, float("inf"), [1])
    names.assert_called_once_with(wh, 1)
    assert wh.relative.call_args.kwargs == {"allow_link": True}
    wh.set_info.assert_called_once()
    wh.close.assert_called_once_with(2)


def test_cleanup_uses_retained_handle_for_directory_enumeration():
    wh = fake_handles(0x10)
    with patch.object(
        gate, "directory_names", side_effect=[iter(["child"]), iter([])]
    ) as names:
        gate.remove_contents(wh, 1, float("inf"), [1])
    assert [call.args[1] for call in names.call_args_list] == [1, 2]
    wh.set_info.assert_called_once()
    wh.close.assert_called_once_with(2)


@pytest.mark.parametrize("deadline,budget", [(0, 10), (float("inf"), 0)])
def test_cleanup_bounds_precede_child_open(deadline, budget):
    wh = fake_handles(0)
    with patch.object(
        gate, "directory_names", return_value=iter(["child"])
    ), pytest.raises(RuntimeError, match="cleanup_budget"):
        gate.remove_contents(wh, 1, deadline, [budget])
    wh.relative.assert_not_called()


def test_cleanup_failure_closes_owned_handle():
    wh = fake_handles(0)
    wh.set_info.return_value = False
    with patch.object(
        gate, "directory_names", return_value=iter(["child"])
    ), pytest.raises(RuntimeError, match="cleanup_delete"):
        gate.remove_contents(wh, 1, float("inf"), [1])
    wh.close.assert_called_once_with(2)


def test_cleanup_retries_transient_kernel_handle_release():
    error = OSError("sharing")
    error.winerror = 32
    operation = Mock(side_effect=[error, 5])
    with patch.object(gate.time, "sleep"):
        assert gate.retry_sharing(operation, float("inf")) == 5
    assert operation.call_count == 2


@pytest.mark.parametrize("code,deadline", [(32, 0), (5, float("inf"))])
def test_cleanup_retry_preserves_deadline_and_other_errors(code, deadline):
    error = OSError("failure")
    error.winerror = code
    operation = Mock(side_effect=error)
    with pytest.raises(OSError):
        gate.retry_sharing(operation, deadline)
    operation.assert_called_once()


def test_directory_record_rejects_escape_component():
    def read(handle, kind, buffer, size):
        record = bytearray(size)
        name = "../outside".encode("utf-16-le")
        struct.pack_into("<I", record, 60, len(name))
        record[104 : 104 + len(name)] = name
        ctypes.memmove(buffer, bytes(record), size)
        return True

    with pytest.raises(RuntimeError, match="invalid_directory_name"):
        list(gate.directory_names(SimpleNamespace(get_info=read), 1))


@pytest.mark.parametrize(
    "test_ok,cleanup_ok,expected", [(True, True, 0), (False, True, 1), (True, False, 1)]
)
def test_main_preserves_test_and_cleanup_failures(
    tmp_path, test_ok, cleanup_ok, expected
):
    report = passing_report()
    if not test_ok:
        report["skipped"] = 1
    (tmp_path / "worker.json").write_text(json.dumps(report))
    (tmp_path / "pytest.log").write_text("test diagnostics")
    fixture = Mock(path=tmp_path)
    if not cleanup_ok:
        fixture.cleanup.side_effect = OSError("injected cleanup")
    worker = {
        "exit_code": 0,
        "timed_out": False,
        "forced_cleanup": False,
        "active_processes": 0,
    }
    with patch.object(gate, "require_native"), patch.object(
        gate, "PrivateFixture", return_value=fixture
    ), patch.object(
        gate, "source_hashes", return_value=report["sources"]
    ), patch.object(
        gate, "supervise", return_value=worker
    ):
        code = gate.main(["--result", str(tmp_path / "result.json")])
    assert code == expected
    saved = json.loads((tmp_path / "result.json").read_text())
    assert saved["cleanup"] == ("passed" if cleanup_ok else "failed")
    fixture.cleanup.assert_called_once()


def test_unproven_process_cleanup_retains_fixture(tmp_path):
    fixture = Mock(path=tmp_path)
    with patch.object(gate, "require_native"), patch.object(
        gate, "PrivateFixture", return_value=fixture
    ), patch.object(gate, "source_hashes", return_value={}), patch.object(
        gate, "supervise", side_effect=RuntimeError("owned_processes_survived")
    ):
        assert gate.main(["--result", str(tmp_path / "result.json")]) == 1
    fixture.cleanup.assert_not_called()
    assert json.loads((tmp_path / "result.json").read_text())["cleanup"] == "failed"


def test_unsafe_parent_rejected_without_worker_or_cleanup(tmp_path):
    with patch.object(gate, "require_native"), patch.object(
        gate, "PrivateFixture", side_effect=PermissionError("unsafe ancestry")
    ), patch.object(gate, "source_hashes", return_value={}), patch.object(
        gate, "supervise"
    ) as supervise:
        assert gate.main(["--result", str(tmp_path / "result.json")]) == 1
    supervise.assert_not_called()
    assert (
        json.loads((tmp_path / "result.json").read_text())["cleanup"] == "not_created"
    )
