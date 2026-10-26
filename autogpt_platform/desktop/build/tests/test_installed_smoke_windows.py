"""Installer-driver boundaries, without executing an installer or runtime."""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

BUILD = Path(__file__).resolve().parents[1]
COMMIT = "a" * 40


@pytest.fixture
def driver(monkeypatch):
    monkeypatch.syspath_prepend(str(BUILD))
    return importlib.import_module("installed_smoke_windows")


def test_provenance_rejects_pr_head_instead_of_expected_merge_commit(driver, tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "build-inputs.json").write_text(
        json.dumps({"commit": "b" * 40, "variant": "experimental"}), encoding="utf-8"
    )
    with pytest.raises(driver.SmokeError, match="artifact_commit"):
        driver.select_artifact(artifact, COMMIT)


@pytest.fixture
def artifact(driver, tmp_path):
    directory = tmp_path / "artifact"
    directory.mkdir()
    desktop = tmp_path / "source/desktop"
    for name in driver.INPUTS:
        path = desktop / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
    record = {
        "commit": COMMIT,
        "variant": "experimental",
        "version": "0.0.0-dev.123",
        "runner": "Windows-2025Server-10.0.26100-SP0",
        "sha256": {name: driver.sha256(desktop / name) for name in driver.INPUTS},
    }
    (directory / "build-inputs.json").write_text(json.dumps(record), encoding="utf-8")
    installer = directory / "AutoGPT-experimental-Setup-0.0.0-dev.123-x64.exe"
    installer.write_bytes(b"MZfixture, never execute")
    return directory, desktop, record, installer


def test_exact_same_run_artifact_and_source_hashes(driver, artifact):
    directory, desktop, record, installer = artifact
    selected, proof = driver.select_artifact(directory, COMMIT, desktop)
    assert selected == installer
    assert proof["commit"] == COMMIT
    assert proof["source_inputs_sha256"] == record["sha256"]
    assert proof["installer_sha256"] == driver.sha256(installer)


@pytest.mark.parametrize(
    "field,value,code",
    [
        ("variant", "", "artifact_variant"),
        ("variant", "experimental-other", "artifact_variant"),
        ("version", "../../escape", "artifact_version"),
        ("version", "0.0.0-dev.01", "artifact_version"),
        ("runner", "Darwin-24", "artifact_os"),
        ("sha256", {}, "build_inputs"),
        ("sha256", {"../../private.env": "a" * 64}, "build_inputs"),
    ],
)
def test_invalid_artifact_metadata(driver, artifact, field, value, code):
    directory, desktop, record, _ = artifact
    record[field] = value
    (directory / "build-inputs.json").write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(driver.SmokeError, match=code):
        driver.select_artifact(directory, COMMIT, desktop)


@pytest.mark.parametrize("extra", ["other.exe", "nested", "build-inputs-copy.json"])
def test_ambiguous_or_unexpected_artifact_paths(driver, artifact, extra):
    directory, desktop, _, _ = artifact
    (directory / extra).mkdir()
    with pytest.raises(driver.SmokeError, match="ambiguous_artifact"):
        driver.select_artifact(directory, COMMIT, desktop)


def test_source_drift_and_non_pe_fail_closed(driver, artifact):
    directory, desktop, _, installer = artifact
    installer.write_bytes(b"script")
    with pytest.raises(driver.SmokeError, match="installer_not_pe"):
        driver.select_artifact(directory, COMMIT, desktop)
    installer.write_bytes(b"MZfixture")
    (desktop / "package.json").write_text("changed", encoding="utf-8")
    with pytest.raises(driver.SmokeError, match="source_inputs"):
        driver.select_artifact(directory, COMMIT, desktop)


def test_duplicate_json_keys_are_not_accepted(driver, tmp_path):
    path = tmp_path / "record.json"
    path.write_text('{"commit":"bad","commit":"good"}', encoding="utf-8")
    with pytest.raises(driver.SmokeError, match="duplicate_json_key"):
        driver.json_object(path)


@pytest.mark.parametrize(
    "path",
    [
        "relative",
        "C:relative",
        r"\\server\share\artifact",
        r"\\?\C:\artifact",
        r"C:\artifact\..\outside",
        r"C:\artifact\file:stream",
        'C:\\artifact\\quote"',
        "C:\\artifact\\trailing. ",
    ],
)
def test_unsafe_windows_path_spellings(driver, path):
    with pytest.raises(driver.SmokeError, match="unsafe_windows_path"):
        driver.windows_path(path)


def test_new_workspace_and_private_directories(driver, tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    work = tmp_path / "new work with spaces"
    private = driver.prepare_work_dir(work, artifact)
    assert (private / "data").is_dir()
    assert list((private / "data").iterdir()) == []
    with pytest.raises(driver.SmokeError, match="work_directory_must_be_new"):
        driver.prepare_work_dir(work, artifact)
    with pytest.raises(driver.SmokeError, match="overlaps_input"):
        driver.prepare_work_dir(artifact / "inside", artifact)


def test_reparse_component_is_refused(driver, tmp_path, monkeypatch):
    original = Path.lstat

    def reparse(path):
        if path == tmp_path:
            return SimpleNamespace(st_mode=0o40700, st_file_attributes=0x400)
        return original(path)

    monkeypatch.setattr(Path, "lstat", reparse)
    with pytest.raises(driver.SmokeError, match="reparse_path"):
        driver.plain_path(tmp_path / "child")


def test_nsis_destination_is_last_and_unquoted_even_with_spaces(driver):
    installer = Path(r"C:\artifact files\installer.exe")
    target = Path(r"C:\private work\install")
    command = driver.installer_command(installer, target)
    assert command == f'"{installer}" /S /currentuser /D={target}'
    assert '"/D=' not in command
    assert "--force-run" not in command


@pytest.fixture
def installed(driver, tmp_path):
    install = tmp_path / "install"
    runtime = install / "resources/runtime"
    (runtime / "python").mkdir(parents=True)
    (install / "autogpt-experimental.exe").write_bytes(b"MZshell")
    (runtime / "python/python.exe").write_bytes(b"MZpython")
    manifest = {
        "win32": {
            "command": "python/python.exe",
            "args": ["-B", "-m", "autogpt_desktop", "serve"],
        },
        "default": {
            "command": "python/bin/python3",
            "args": ["-B", "-m", "autogpt_desktop", "serve"],
        },
    }
    (runtime / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return install, runtime, manifest


@pytest.mark.parametrize(
    "entry",
    [
        {"command": "../../outside.exe", "args": []},
        {"command": "python/python.exe", "args": ["-c", "print('bypass')"]},
        {"command": "C:/host/python.exe", "args": []},
    ],
)
def test_manifest_cannot_escape_or_change_runtime_command(driver, installed, entry):
    install, runtime, manifest = installed
    manifest["win32"] = entry
    (runtime / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(driver.SmokeError, match="unexpected_runtime_manifest"):
        driver.installed_runtime(install)


def test_installed_python_and_full_lifecycle_command(driver, installed, tmp_path):
    install, runtime, _ = installed
    found, python, proof = driver.installed_runtime(install)
    assert found == runtime
    assert python == runtime / "python/python.exe"
    data = tmp_path / "private/data"
    command = driver.smoke_command(python, runtime, data)
    assert command == [
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
    assert proof["python_sha256"] == driver.sha256(python)


def test_environment_excludes_credentials_overrides_and_host_tools(driver, tmp_path):
    system = tmp_path / "Windows"
    powershell = system / "System32/WindowsPowerShell/v1.0/powershell.exe"
    powershell.parent.mkdir(parents=True)
    powershell.write_bytes(b"fixture")
    contaminated = {
        "SystemRoot": str(system),
        "PATH": r"D:\hostedtoolcache\windows\Python;C:\hostmanagedtools",
        "PYTHONPATH": "external-packages",
        "PYTHONHOME": "host-python",
        "NODE_OPTIONS": "--require malicious.js",
        "AUTOGPT_DESKTOP_NODE": "host-node",
        "AUTOGPT_DESKTOP_RUNTIME": "external-runtime",
        "AWS_SECRET_ACCESS_KEY": "private-sentinel",
        "GITHUB_TOKEN": "private-sentinel",
        "OPENAI_API_KEY": "private-sentinel",
        "CLAUDE_CODE_OAUTH_TOKEN": "private-sentinel",
        "HOME": "personal-profile",
        "APPDATA": "personal-settings",
    }
    env = driver.private_environment(tmp_path / "private", contaminated)
    assert "private-sentinel" not in json.dumps(env)
    assert "personal-" not in json.dumps(env)
    assert "hostmanagedtools" not in env["PATH"]
    assert "hostedtoolcache" not in env["PATH"]
    assert set(contaminated).intersection(env) == {"PATH", "HOME", "APPDATA"}
    assert env["AUTOGPT_CLAUDE_CODE"] == "off"
    assert env["CHAT_TEST_MODE"] == "true"
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["DISABLE_ERROR_REPORTING"] == "1"


@pytest.mark.parametrize(
    "starts,success,expected",
    [
        ([1, 2, 3, 4], True, True),
        ([1], True, False),
        ([1, 2, 3, 4], False, False),
        ([1, 2, 2, 3, 4], True, False),
    ],
)
def test_safe_summary_requires_all_four_starts(
    driver, tmp_path, starts, success, expected
):
    log = tmp_path / "raw.log"
    log.write_text(
        "secret-token=do-not-upload\n"
        + "\n".join(f"== start {n} of 4: {driver.STARTS[n - 1]}" for n in starts)
        + ("\nOK in 300s\n" if success else "\nFAILED\n"),
        encoding="utf-8",
    )
    summary = driver.lifecycle_summary(log)
    assert summary["completed"] is expected
    assert "do-not-upload" not in json.dumps(summary)


def diagnostic_record(**updates):
    return {
        "version": 1,
        "stage": "backend_contract",
        "status": "failed",
        "reason": "child_exit",
        "exception": "none",
        "contract": "none",
        "child_exit": 1,
        "failures": 1,
        "start": 0,
        **updates,
    }


def runtime_record(**updates):
    return {
        "version": 1,
        "start": 1,
        "point": "ready",
        "progress": "services",
        "phase": "database",
        "fatal": True,
        "error": "process_exited_starting",
        "process": "workers",
        "process_exit": None,
        "os_error": "none",
        "stop": "not_observed",
        "stop_exit": None,
        "settle": "fatal_event",
        "elapsed_ms": 21000,
        "completed_phases": 3,
        "runtime_exit": None,
        "events_rejected": False,
        "events_truncated": False,
        **updates,
    }


def test_runtime_context_summary_is_separate_and_never_completes(driver, tmp_path):
    log = tmp_path / "raw.log"
    log.write_text(
        "PRIVATE-SENTINEL\nDESKTOP_SMOKE_RUNTIME_DIAGNOSTIC "
        + json.dumps(runtime_record())
        + "\n"
    )
    extra = {}
    lifecycle = driver.lifecycle_summary(log, runtime_diagnostics=extra)
    assert set(lifecycle) == {
        "starts",
        "completed",
        "diagnostics",
        "diagnostics_rejected",
        "truncated",
    }
    assert not lifecycle["completed"]
    assert extra == {
        "records": [runtime_record()],
        "rejected": False,
        "truncated": False,
    }
    assert "PRIVATE-SENTINEL" not in json.dumps(extra)


@pytest.mark.parametrize(
    "updates",
    [
        {"version": True},
        {"version": 2},
        {"start": True},
        {"start": 0},
        {"start": 5},
        {"point": "PRIVATE-SENTINEL"},
        {"progress": []},
        {"phase": "PRIVATE-SENTINEL"},
        {"fatal": 1},
        {"error": "PRIVATE-SENTINEL"},
        {"process": "PRIVATE-SENTINEL"},
        {"process_exit": True},
        {"process_exit": -2147483649},
        {"process_exit": 4294967296},
        {"stop_exit": 0.0},
        {"stop_exit": True},
        {"os_error": "PRIVATE-SENTINEL"},
        {"stop": "PRIVATE-SENTINEL"},
        {"events_rejected": 0},
        {"events_truncated": []},
        {"message": "PRIVATE-SENTINEL"},
        {"settle": "PRIVATE-SENTINEL"},
        {"elapsed_ms": True},
        {"elapsed_ms": -1},
        {"elapsed_ms": 3600001},
        {"completed_phases": 128},
        {"completed_phases": None},
        {"runtime_exit": True},
    ],
)
def test_runtime_context_consumer_rejects_untrusted_records(driver, tmp_path, updates):
    log = tmp_path / "raw.log"
    log.write_text(
        "DESKTOP_SMOKE_RUNTIME_DIAGNOSTIC "
        + json.dumps(runtime_record(**updates))
        + "\n"
    )
    extra = {}
    lifecycle = driver.lifecycle_summary(log, runtime_diagnostics=extra)
    assert extra == {"records": [], "rejected": True, "truncated": False}
    assert lifecycle["diagnostics"] == []
    assert not lifecycle["completed"]
    assert "PRIVATE-SENTINEL" not in json.dumps(extra)


@pytest.mark.parametrize(
    "payload", ["[]", "{", '{"version":1,"version":1}', " " * 2049]
)
def test_runtime_context_consumer_rejects_bad_json_and_size(driver, tmp_path, payload):
    log = tmp_path / "raw.log"
    log.write_text("DESKTOP_SMOKE_RUNTIME_DIAGNOSTIC " + payload + "\n")
    extra = {}
    driver.lifecycle_summary(log, runtime_diagnostics=extra)
    assert extra["rejected"] and not extra["records"]


def test_runtime_context_consumer_caps_records_without_skipping_lifecycle(
    driver, tmp_path
):
    log = tmp_path / "raw.log"
    log.write_text(
        ("DESKTOP_SMOKE_RUNTIME_DIAGNOSTIC " + json.dumps(runtime_record()) + "\n") * 9
        + "\n".join(
            f"== start {i} of 4: {name}" for i, name in enumerate(driver.STARTS, 1)
        )
        + "\nOK in 1s\n"
    )
    extra = {}
    lifecycle = driver.lifecycle_summary(log, runtime_diagnostics=extra)
    assert len(extra["records"]) == 8
    assert extra["truncated"]
    assert lifecycle["completed"]


def test_runtime_context_consumer_rejects_duplicate_in_otherwise_valid_record(
    driver, tmp_path
):
    log = tmp_path / "raw.log"
    payload = json.dumps(runtime_record()).replace(
        '"version": 1', '"version": 1, "version": 1'
    )
    log.write_text("DESKTOP_SMOKE_RUNTIME_DIAGNOSTIC " + payload + "\n")
    extra = {}
    driver.lifecycle_summary(log, runtime_diagnostics=extra)
    assert extra == {"records": [], "rejected": True, "truncated": False}


def test_runtime_context_overlong_line_cannot_smuggle_record(driver, tmp_path):
    log = tmp_path / "raw.log"
    record = "DESKTOP_SMOKE_RUNTIME_DIAGNOSTIC " + json.dumps(runtime_record())
    log.write_text(
        "DESKTOP_SMOKE_RUNTIME_DIAGNOSTIC "
        + "x" * 17000
        + record
        + "\n"
        + record
        + "\n"
    )
    extra = {}
    driver.lifecycle_summary(log, runtime_diagnostics=extra)
    assert extra == {
        "records": [runtime_record()],
        "rejected": True,
        "truncated": False,
    }


def test_runtime_context_absence_preserves_legacy_packet(driver, tmp_path):
    log = tmp_path / "raw.log"
    log.write_text("FAILED\n")
    extra = {}
    assert driver.lifecycle_summary(log, runtime_diagnostics=extra) == {
        "starts": [],
        "completed": False,
        "diagnostics": [],
        "diagnostics_rejected": False,
        "truncated": False,
    }
    assert extra == {}


def test_safe_failure_diagnostics_survive_without_private_output(driver, tmp_path):
    log = tmp_path / "raw.log"
    record = diagnostic_record()
    log.write_text(
        "PRIVATE-SENTINEL\nDESKTOP_SMOKE_DIAGNOSTIC "
        + json.dumps(record)
        + "\nFAILED\nPRIVATE-SENTINEL\n",
        encoding="utf-8",
    )
    summary = driver.lifecycle_summary(log)
    assert summary["diagnostics"] == [record]
    assert not summary["completed"]
    assert "PRIVATE-SENTINEL" not in json.dumps(summary)


@pytest.mark.parametrize(
    "updates",
    [
        {"version": True},
        {"version": 2},
        {"stage": "PRIVATE-SENTINEL"},
        {"status": "PRIVATE-SENTINEL"},
        {"reason": "PRIVATE-SENTINEL"},
        {"exception": "PRIVATE-SENTINEL"},
        {"contract": "PRIVATE-SENTINEL"},
        {"stage": ["smoke_result"]},
        {"child_exit": True},
        {"child_exit": 4294967296},
        {"child_exit": -2147483649},
        {"child_exit": 1.0},
        {"failures": True},
        {"failures": 65536},
        {"failures": -1},
        {"start": True},
        {"start": 5},
        {"extra": "PRIVATE-SENTINEL"},
    ],
)
def test_invalid_diagnostic_is_never_uploaded_or_treated_as_success(
    driver, tmp_path, updates
):
    log = tmp_path / "raw.log"
    log.write_text(
        "DESKTOP_SMOKE_DIAGNOSTIC "
        + json.dumps(diagnostic_record(**updates))
        + "\n"
        + "\n".join(
            f"== start {i} of 4: {name}" for i, name in enumerate(driver.STARTS, 1)
        )
        + "\nOK in 1s\n",
        encoding="utf-8",
    )
    summary = driver.lifecycle_summary(log)
    assert summary["diagnostics"] == []
    assert summary["diagnostics_rejected"]
    assert not summary["completed"]
    assert "PRIVATE-SENTINEL" not in json.dumps(summary)


@pytest.mark.parametrize("payload", ['{"version":1,"version":1}', "[]", "{"])
def test_malformed_or_duplicate_diagnostic_json_is_rejected(driver, tmp_path, payload):
    log = tmp_path / "raw.log"
    log.write_text("DESKTOP_SMOKE_DIAGNOSTIC " + payload + "\n", encoding="utf-8")
    summary = driver.lifecycle_summary(log)
    assert summary["diagnostics"] == []
    assert summary["diagnostics_rejected"]


def test_summary_bounds_cannot_hide_later_contradictory_evidence(
    driver, tmp_path, monkeypatch
):
    log = tmp_path / "raw.log"
    success = (
        "\n".join(
            f"== start {i} of 4: {name}" for i, name in enumerate(driver.STARTS, 1)
        )
        + "\nOK in 1s\n"
    )
    log.write_text(success + "PRIVATE-SENTINEL\n" * 100, encoding="utf-8")
    monkeypatch.setattr(driver, "MAX_SUMMARY_BYTES", len(success.encode()) + 1)
    summary = driver.lifecycle_summary(log)
    assert summary["truncated"]
    assert not summary["completed"]
    assert "PRIVATE-SENTINEL" not in json.dumps(summary)


def test_oversized_line_cannot_forge_a_new_line_diagnostic(
    driver, tmp_path, monkeypatch
):
    log = tmp_path / "raw.log"
    record = diagnostic_record()
    payload = "DESKTOP_SMOKE_DIAGNOSTIC " + json.dumps(record)
    monkeypatch.setattr(driver, "MAX_SUMMARY_LINE_BYTES", len(payload.encode()) + 10)
    log.write_text(
        "x" * (len(payload) + 11) + payload + "\n" + payload, encoding="utf-8"
    )
    summary = driver.lifecycle_summary(log)
    assert summary["diagnostics"] == [record]
    assert "PRIVATE-SENTINEL" not in json.dumps(summary)


def test_diagnostic_events_are_bounded_and_cannot_establish_completion(
    driver, tmp_path, monkeypatch
):
    log = tmp_path / "raw.log"
    monkeypatch.setattr(driver, "MAX_DIAGNOSTICS", 2)
    record = diagnostic_record(stage="smoke_result", status="passed", reason="none")
    log.write_text(
        ("DESKTOP_SMOKE_DIAGNOSTIC " + json.dumps(record) + "\n") * 3,
        encoding="utf-8",
    )
    summary = driver.lifecycle_summary(log)
    assert summary["diagnostics"] == [record, record]
    assert summary["truncated"]
    assert not summary["completed"]


@pytest.fixture
def fake_stage(driver, tmp_path, monkeypatch):
    private = tmp_path / "private"
    (private / "logs").mkdir(parents=True)
    clock = [0.0]
    monkeypatch.setattr(driver.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        driver.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )
    calls = []
    process = SimpleNamespace(pid=123, returncode=None)
    process.poll = lambda: process.returncode

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return process

    monkeypatch.setattr(driver.subprocess, "Popen", popen)
    owned = SimpleNamespace(
        psutil=SimpleNamespace(
            NoSuchProcess=ProcessLookupError, Process=lambda pid: pid
        ),
        remember=lambda *args: None,
        observe=lambda: None,
    )
    return private, calls, process, owned


@pytest.mark.parametrize("stage", ["install", "lifecycle"])
def test_stage_timeout_is_bounded_and_not_success(driver, fake_stage, stage):
    private, calls, _, owned = fake_stage
    stages = []
    with pytest.raises(driver.SmokeError, match=f"{stage}_timeout"):
        driver.run_stage(stage, ["never-execute"], 1, private, {}, owned, stages)
    assert stages[0]["timed_out"]
    assert stages[0]["elapsed_seconds"] == 1
    assert calls[0][1]["shell"] is False
    assert calls[0][1]["stdin"] is subprocess.DEVNULL
    assert calls[0][1]["stderr"] is subprocess.STDOUT


@pytest.mark.parametrize("exit_code", [1, -9, 0xC0000135])
def test_negative_lifecycle_exit_is_never_passed(driver, fake_stage, exit_code):
    private, _, process, owned = fake_stage
    process.returncode = exit_code
    stages = []
    with pytest.raises(driver.SmokeError, match="lifecycle_exit"):
        driver.run_stage("lifecycle", ["never-execute"], 10, private, {}, owned, stages)
    assert stages[0]["exit_code"] == exit_code


def test_non_windows_and_non_hosted_hosts_refused_without_processes(
    driver, monkeypatch
):
    monkeypatch.setattr(driver, "WINDOWS", False)
    with pytest.raises(driver.SmokeError, match="windows_required"):
        driver.require_clean_host()
    monkeypatch.setattr(driver, "WINDOWS", True)
    monkeypatch.setattr(os, "environ", {})
    with pytest.raises(driver.SmokeError, match="github_hosted_runner"):
        driver.require_clean_host()


@pytest.fixture
def processes(driver, tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(driver.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(driver.time, "time", lambda: 100.0)
    monkeypatch.setattr(
        driver.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )
    members = {}
    killed = []

    class ProcessError(Exception):
        pass

    class Gone(ProcessError):
        pass

    class Process:
        def __init__(
            self, pid, born=101.0, executable=None, children=(), stubborn=False
        ):
            self.pid = pid
            self.born = born
            self.executable = str(executable or tmp_path / "outside.exe")
            self.descendants = children
            self.alive = True
            self.stubborn = stubborn
            members[pid] = self

        def create_time(self):
            return self.born

        def is_running(self):
            return self.alive

        def children(self, recursive=False):
            return [child for child in self.descendants if child.alive]

        def exe(self):
            return self.executable

        def net_connections(self, kind):
            return []

        def kill(self):
            killed.append(self.pid)
            if not self.stubborn:
                self.alive = False

    def lookup(pid):
        if pid not in members or not members[pid].alive:
            raise Gone()
        return members[pid]

    psutil = SimpleNamespace(
        Process=lookup,
        Error=ProcessError,
        NoSuchProcess=Gone,
        process_iter=lambda: list(members.values()),
        net_connections=lambda kind: [],
        CONN_LISTEN="LISTEN",
    )
    monkeypatch.setitem(sys.modules, "psutil", psutil)
    install = tmp_path / "install"
    data = tmp_path / "data"
    tracker = driver.OwnedProcesses(install, data)
    return SimpleNamespace(
        tracker=tracker,
        Process=Process,
        killed=killed,
        members=members,
        psutil=psutil,
        install=install,
        data=data,
        clock=clock,
    )


def test_cleanup_only_observed_tree_or_new_exclusive_install_processes(
    driver, processes
):
    p = processes
    child = p.Process(2)
    parent = p.Process(1, children=[child])
    p.tracker.remember(parent, "launched_stage")
    detached = p.Process(3, executable=p.install / "postgres/postgres.exe")
    unrelated = p.Process(
        4, executable=p.install.with_name("install-other") / "python.exe"
    )
    older = p.Process(5, born=90, executable=p.install / "old.exe")
    report = p.tracker.cleanup()
    assert set(p.killed) == {1, 2, 3}
    assert unrelated.alive and older.alive and not detached.alive
    assert report["surviving_owned_pids"] == []
    assert {entry["provenance"] for entry in report["observed_processes"]} == {
        "launched_stage",
        "observed_descendant",
        "new_install_executable",
    }


def test_reused_pid_is_not_killed(driver, processes):
    p = processes
    original = p.Process(1, born=100)
    p.tracker.remember(original, "launched_stage")
    replacement = p.Process(1, born=102)
    report = p.tracker.cleanup()
    assert replacement.alive
    assert p.killed == []
    assert report["surviving_owned_pids"] == []


def test_cleanup_rescans_after_last_parent_dies(driver, processes):
    p = processes
    parent = p.Process(1)
    p.tracker.remember(parent, "launched_stage")

    def kill_and_spawn():
        p.killed.append(1)
        parent.alive = False
        p.Process(2, executable=p.install / "late-child.exe")

    parent.kill = kill_and_spawn
    report = p.tracker.cleanup()
    assert set(p.killed) == {1, 2}
    assert report["surviving_owned_pids"] == []


@pytest.mark.parametrize("hive", [1, 2])
@pytest.mark.parametrize("view", [32, 64])
@pytest.mark.parametrize("key_index", [0, 1])
def test_exact_installer_identity_refused_even_without_display_name(
    driver, monkeypatch, hive, view, key_index
):
    monkeypatch.setattr(driver, "WINDOWS", True)
    monkeypatch.setattr(driver.sys, "version_info", (3, 13))
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("RUNNER_ENVIRONMENT", "github-hosted")

    class Key:
        values = {"UninstallString": r"C:\existing\Uninstall.exe"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def open_key(found_hive, path, reserved, access):
        if (found_hive, path, access) == (hive, driver.INSTALL_KEYS[key_index], view):
            return Key()
        raise FileNotFoundError()

    monkeypatch.setitem(
        sys.modules,
        "winreg",
        SimpleNamespace(
            HKEY_CURRENT_USER=1,
            HKEY_LOCAL_MACHINE=2,
            KEY_WOW64_32KEY=32,
            KEY_WOW64_64KEY=64,
            KEY_READ=0,
            OpenKey=open_key,
        ),
    )
    with pytest.raises(driver.SmokeError, match="existing_experimental_install"):
        driver.require_clean_host()


def test_spaced_private_path_without_short_names_is_refused_before_install(
    driver, monkeypatch
):
    monkeypatch.setattr(driver, "short_path", lambda value: value)
    with pytest.raises(driver.SmokeError, match="space_free_short_name"):
        driver.private_environment(Path(r"C:\private smoke\home"), {})
    monkeypatch.setattr(driver, "short_path", lambda value: r"C:\PRIVAT~1\home")
    driver.require_alias_root(Path(r"C:\private smoke\home"))


def test_allocated_ports_checked_even_when_no_owned_process_remains(
    driver, processes, monkeypatch
):
    p = processes
    (p.data / "config").mkdir(parents=True)
    ports = p.data / "config/ports.json"
    ports.write_text(json.dumps({"public": 18473, "postgres": 15432}), encoding="utf-8")
    p.tracker.observe()
    ports.write_text(json.dumps({"public": 18474, "postgres": 15432}), encoding="utf-8")
    unrelated = p.Process(20)
    p.psutil.net_connections = lambda kind: [
        SimpleNamespace(status="LISTEN", laddr=SimpleNamespace(port=18473), pid=20)
    ]
    report = p.tracker.cleanup()
    assert report["observed_ports"] == [15432, 18473, 18474]
    assert report["ports_still_open"] == [18473]
    assert report["remaining_listeners"] == [{"port": 18473, "pid": 20, "owned": False}]
    assert unrelated.alive and p.killed == []


@pytest.mark.parametrize(
    "value", [{"public": "secret-value"}, {"public": True}, {"public": 65536}]
)
def test_invalid_final_port_record_fails_proof(driver, processes, value):
    p = processes
    (p.data / "config").mkdir(parents=True)
    (p.data / "config/ports.json").write_text(json.dumps(value), encoding="utf-8")
    report = p.tracker.cleanup()
    assert report["inspection_failed"]
    assert "secret-value" not in json.dumps(report)


def test_stubborn_owned_process_does_not_make_cleanup_unbounded(driver, processes):
    p = processes
    process = p.Process(10, stubborn=True)
    p.tracker.remember(process, "launched_stage")
    report = p.tracker.cleanup(timeout=20)
    assert 5 <= p.clock[0] < 20
    assert report["surviving_owned_pids"] == [10]


@pytest.mark.parametrize("timeout", [0.05, 1, 15])
def test_short_cleanup_budget_attempts_termination(processes, timeout):
    p = processes
    process = p.Process(12)
    p.tracker.remember(process, "launched_stage")
    report = p.tracker.cleanup(timeout=timeout)
    assert p.killed == [12]
    assert not process.alive
    assert p.clock[0] <= timeout
    assert report["surviving_owned_pids"] == []


def test_cleanup_latches_listener_query_deadline_overrun(processes):
    p = processes
    p.tracker.ports.add(18473)

    def slow_listener_table(kind):
        p.clock[0] += 121
        return []

    p.psutil.net_connections = slow_listener_table
    report = p.tracker.cleanup(timeout=120)
    assert report["inspection_failed"]
    assert report["deadline_exceeded"]
    assert report["elapsed_seconds"] == 121


@pytest.mark.parametrize("leak", [False, True])
def test_main_late_cleanup_discovery_cannot_hide_lifecycle_leak(
    driver, processes, artifact, monkeypatch, tmp_path, leak
):
    p = processes
    directory, desktop, _, _ = artifact
    work = tmp_path / "work"
    monkeypatch.setattr(driver, "require_clean_host", lambda: None)
    monkeypatch.setenv("GITHUB_SHA", COMMIT)
    select = driver.select_artifact
    monkeypatch.setattr(
        driver, "select_artifact", lambda folder, sha: select(folder, sha, desktop)
    )
    monkeypatch.setattr(driver, "private_environment", lambda *args: {})
    monkeypatch.setattr(driver, "OwnedProcesses", lambda *args: p.tracker)
    runtime = p.install / "resources/runtime"
    monkeypatch.setattr(
        driver, "installed_runtime", lambda path: (runtime, runtime / "python.exe", {})
    )

    def stage(name, command, timeout, private, env, tracker, stages):
        stages.append({"name": name, "exit_code": 0})
        if name != "lifecycle":
            return
        (private / "logs/lifecycle.log").write_text(
            "token=PRIVATE-SENTINEL\n"
            + "\n".join(
                f"== start {index} of 4: {title}"
                for index, title in enumerate(driver.STARTS, 1)
            )
            + "\nOK in 1s\n",
            encoding="utf-8",
        )
        (p.data / "config").mkdir(parents=True)
        (p.data / "config/ports.json").write_text('{"public":18473}', encoding="utf-8")
        if leak:
            child = p.Process(55, executable=p.install / "runtime-worker.exe")
            executable = child.exe
            reads = 0

            def briefly_uninspectable():
                nonlocal reads
                reads += 1
                if reads == 1:
                    raise p.psutil.Error("synthetic transient inspection failure")
                return executable()

            child.exe = briefly_uninspectable

    monkeypatch.setattr(driver, "run_stage", stage)
    code = driver.main(
        [
            "--artifact-dir",
            str(directory),
            "--work-dir",
            str(work),
            "--expected-commit",
            COMMIT,
        ]
    )
    result = json.loads((work / "result.json").read_text("utf-8"))
    assert p.killed == ([55] if leak else [])
    cleanup = result["cleanup"]
    assert cleanup["leftovers_before_cleanup"] == 0
    assert cleanup["surviving_owned_pids"] == []
    assert code == int(leak)
    assert result["ok"] is not leak
    assert cleanup["live_owned_during_cleanup"] == (
        [{"pid": 55, "created": 101.0}] if leak else []
    )
    assert result.get("error") == ("owned_process_or_port_leak" if leak else None)
    assert "PRIVATE-SENTINEL" not in json.dumps(result)
    assert result["upload_allowlist"] == ["result.json"]


@pytest.mark.parametrize(
    "failure",
    [
        "install_timeout",
        "lifecycle_exit",
        "lifecycle_timeout",
        "driver_exception_private_logs_only",
        "incomplete_lifecycle",
        None,
    ],
)
@pytest.mark.parametrize("unreadable", [False, True])
def test_main_failure_cleans_up_and_publishes_only_safe_summary(
    driver, artifact, tmp_path, monkeypatch, failure, unreadable
):
    directory, desktop, _, _ = artifact
    work = tmp_path / "new work"
    monkeypatch.setattr(driver, "require_clean_host", lambda: None)
    monkeypatch.setenv("GITHUB_SHA", COMMIT)
    select = driver.select_artifact
    monkeypatch.setattr(
        driver, "select_artifact", lambda folder, sha: select(folder, sha, desktop)
    )
    monkeypatch.setattr(driver, "private_environment", lambda *args: {})
    cleaned = []
    owned = SimpleNamespace(
        observe=lambda: None,
        living=lambda: [],
        cleanup=lambda: (
            cleaned.append(True)
            or {
                "leftovers_before_cleanup": 0,
                "live_owned_during_cleanup": [],
                "surviving_owned_pids": [],
                "ports_still_open": [],
                "ports_not_checked": [],
                "inspection_failed": False,
                "allocated_ports": [18473],
            }
        ),
    )
    monkeypatch.setattr(driver, "OwnedProcesses", lambda *args: owned)
    runtime = tmp_path / "runtime"
    monkeypatch.setattr(
        driver, "installed_runtime", lambda path: (runtime, runtime / "python.exe", {})
    )
    if unreadable:

        def cannot_read(log, **kwargs):
            raise OSError("PRIVATE-SENTINEL")

        monkeypatch.setattr(driver, "lifecycle_summary", cannot_read)

    def stage(name, command, timeout, private, env, tracker, stages):
        stages.append({"name": name, "exit_code": 0})
        headings = (
            "\n".join(
                f"== start {index} of 4: {title}"
                for index, title in enumerate(driver.STARTS, 1)
            )
            if failure is None
            else ""
        )
        (private / "logs" / f"{name}.log").write_text(
            "token=PRIVATE-SENTINEL\n"
            + headings
            + "\nOK in 1s\n"
            + "DESKTOP_SMOKE_DIAGNOSTIC "
            + json.dumps(diagnostic_record())
            + "\nDESKTOP_SMOKE_RUNTIME_DIAGNOSTIC "
            + json.dumps(runtime_record())
            + "\n",
            encoding="utf-8",
        )
        if name == "lifecycle" and failure == "driver_exception_private_logs_only":
            raise RuntimeError("PRIVATE-SENTINEL")
        if failure and failure.startswith(name):
            stages[-1]["exit_code"] = 1
            raise driver.SmokeError(failure)

    monkeypatch.setattr(driver, "run_stage", stage)
    expected_failure = failure or ("incomplete_lifecycle" if unreadable else None)
    assert driver.main(
        [
            "--artifact-dir",
            str(directory),
            "--work-dir",
            str(work),
            "--expected-commit",
            COMMIT,
        ]
    ) == (1 if expected_failure else 0)
    result = json.loads((work / "result.json").read_text("utf-8"))
    assert result.get("error") == expected_failure
    assert result["ok"] is (expected_failure is None)
    if failure != "install_timeout":
        assert result["lifecycle"]["starts"] == (
            [1, 2, 3, 4] if expected_failure is None else []
        )
        assert result["lifecycle"]["completed"] is (expected_failure is None)
        if unreadable:
            assert result["lifecycle"]["summary_unavailable"]
        else:
            assert result["lifecycle"]["diagnostics"] == [diagnostic_record()]
            assert result["runtime_diagnostics"] == {
                "records": [runtime_record()],
                "rejected": False,
                "truncated": False,
            }
            assert "runtime_diagnostics" not in result["lifecycle"]
    assert cleaned == [True]
    assert "PRIVATE-SENTINEL" not in json.dumps(result)
    assert result["upload_allowlist"] == ["result.json"]
