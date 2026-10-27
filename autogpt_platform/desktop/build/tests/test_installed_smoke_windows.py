"""Installer-driver boundaries, without executing an installer or runtime."""

import ast
import importlib
import io
import json
import logging
import os
import subprocess
import sys
import traceback
from pathlib import Path
from types import CodeType, FunctionType, SimpleNamespace

import pytest

BUILD = Path(__file__).resolve().parents[1]
COMMIT = "a" * 40


@pytest.fixture
def driver(monkeypatch):
    monkeypatch.syspath_prepend(str(BUILD))
    monkeypatch.syspath_prepend(str(BUILD.parent / "runtime"))
    from autogpt_desktop import winlinks

    def create_private(path, user_sid):
        path.mkdir()
        return True

    monkeypatch.setattr(winlinks, "current_user_sid", lambda: "S-1-5-21-123")
    monkeypatch.setattr(winlinks, "create_private", create_private)
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


def test_windows_all_eight_roots_use_atomic_private_creation(
    driver, tmp_path, monkeypatch
):
    from autogpt_desktop import winlinks

    monkeypatch.setattr(driver, "WINDOWS", True)
    monkeypatch.setattr(driver, "windows_path", lambda value: None)
    created = []

    def create(path, user_sid):
        created.append((path, user_sid))
        path.mkdir()
        return True

    monkeypatch.setattr(winlinks, "create_private", create)
    work = tmp_path / "new-work"
    private = driver.prepare_work_dir(work, tmp_path / "artifact")
    assert created == [
        (path, "S-1-5-21-123")
        for path in (
            work,
            private,
            *(
                private / name
                for name in ("logs", "home", "appdata", "localappdata", "temp", "data")
            ),
        )
    ]


@pytest.mark.parametrize("raced_index", range(8))
def test_raced_private_root_refused_before_installer(
    driver, tmp_path, monkeypatch, raced_index
):
    from autogpt_desktop import winlinks

    monkeypatch.setattr(driver, "WINDOWS", True)
    monkeypatch.setattr(driver, "windows_path", lambda value: None)
    monkeypatch.setattr(driver, "require_clean_host", lambda: None)
    monkeypatch.setenv("GITHUB_SHA", COMMIT)
    created = []

    def create(path, user_sid):
        path.mkdir()
        created.append(path)
        if len(created) == raced_index + 1:
            (path / "sentinel").write_bytes(b"preserve")
            return False
        return True

    def forbidden(*args, **kwargs):
        pytest.fail("installer or ACL repair reached after a raced root")

    monkeypatch.setattr(winlinks, "create_private", create)
    monkeypatch.setattr(Path, "chmod", forbidden)
    monkeypatch.setattr(driver, "select_artifact", forbidden)
    monkeypatch.setattr(driver, "run_stage", forbidden)
    work = tmp_path / "new-work"
    assert (
        driver.main(
            [
                "--artifact-dir",
                str(tmp_path / "artifact"),
                "--work-dir",
                str(work),
                "--expected-commit",
                COMMIT,
            ]
        )
        == 1
    )
    assert len(created) == raced_index + 1
    assert (created[-1] / "sentinel").read_bytes() == b"preserve"


def test_posix_workspace_keeps_private_modes_without_native_calls(
    driver, tmp_path, monkeypatch
):
    from autogpt_desktop import winlinks

    monkeypatch.setattr(driver, "WINDOWS", False)
    calls = []
    mkdir = Path.mkdir

    def make(path, mode=0o777, **kwargs):
        calls.append(mode)
        return mkdir(path, mode=mode, **kwargs)

    def forbidden(*args):
        pytest.fail("POSIX factory called Windows API")

    monkeypatch.setattr(Path, "mkdir", make)
    monkeypatch.setattr(winlinks, "current_user_sid", forbidden)
    monkeypatch.setattr(winlinks, "create_private", forbidden)
    driver.prepare_work_dir(tmp_path / "work", tmp_path / "artifact")
    assert calls == [0o700] * 8


@pytest.mark.parametrize("problem", ["existing", "overlap", "reparse", "no_parent"])
def test_private_root_checks_precede_native_creation(
    driver, tmp_path, monkeypatch, problem
):
    from autogpt_desktop import winlinks

    monkeypatch.setattr(driver, "WINDOWS", True)
    monkeypatch.setattr(driver, "windows_path", lambda value: None)
    work = tmp_path / "work"
    artifact = tmp_path / "artifact"
    if problem == "existing":
        work.mkdir()
        (work / "sentinel").write_text("preserve")
    elif problem == "overlap":
        artifact = tmp_path
    elif problem == "no_parent":
        work = tmp_path / "missing/work"
    else:
        original = Path.lstat

        def reparse(path):
            if path == tmp_path:
                return SimpleNamespace(st_mode=0o40700, st_file_attributes=0x400)
            return original(path)

        monkeypatch.setattr(Path, "lstat", reparse)

    def forbidden(*args):
        pytest.fail("native creation reached for unsafe work root")

    monkeypatch.setattr(winlinks, "current_user_sid", forbidden)
    monkeypatch.setattr(winlinks, "create_private", forbidden)
    with pytest.raises(driver.SmokeError):
        driver.prepare_work_dir(work, artifact)
    if problem == "existing":
        assert (work / "sentinel").read_text() == "preserve"
        assert list(work.iterdir()) == [work / "sentinel"]
    else:
        assert not work.exists()


def test_token_lookup_failure_creates_no_private_tree(driver, tmp_path, monkeypatch):
    from autogpt_desktop import winlinks

    monkeypatch.setattr(driver, "WINDOWS", True)
    monkeypatch.setattr(driver, "windows_path", lambda value: None)

    def unavailable():
        raise OSError("token unavailable")

    monkeypatch.setattr(winlinks, "current_user_sid", unavailable)
    with pytest.raises(OSError, match="token unavailable"):
        driver.prepare_work_dir(tmp_path / "work", tmp_path / "artifact")
    assert not (tmp_path / "work").exists()


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


TRACE_MARKER = "2026-10-26 12:34:56,789 [desktop] ERROR startup failed\n"
TRACE_HEADER = "Traceback (most recent call last):\n"
TRACE_FRAME = (
    '  File "C:\\PRIVATE-CANARY\\autogpt_desktop\\supervisor.py", '
    "line 210, in PRIVATE_FUNCTION\n"
)
TRACE_BODY = TRACE_HEADER + TRACE_FRAME + "RuntimeError: PRIVATE-MESSAGE\n"


def read_trace(driver, tmp_path, text):
    log = tmp_path / "lifecycle.log"
    log.write_bytes(text.encode("utf-8") if isinstance(text, str) else text)
    return driver.startup_failure_trace(log)


def trace_result(exceptions=(), **flags):
    return {
        "version": 1,
        "exceptions": list(exceptions),
        "truncated": False,
        "rejected": False,
        "unavailable": False,
        **flags,
    }


def trace_exception(category="runtime_error", frames=None, omitted=0):
    return {
        "category": category,
        "frames": (
            [{"module": "autogpt_desktop.supervisor", "line": 210}]
            if frames is None
            else frames
        ),
        "omitted_frames": omitted,
    }


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("chain", ["cause", "context"])
def test_startup_trace_real_python_traceback(driver, tmp_path, newline, chain):
    def trigger():
        try:
            raise FileNotFoundError("PRIVATE-URL https://secret.invalid/token")
        except FileNotFoundError as cause:
            if chain == "cause":
                raise RuntimeError("PRIVATE-MESSAGE") from cause
            raise RuntimeError("PRIVATE-MESSAGE")

    trigger.__code__ = trigger.__code__.replace(
        co_filename="C:/PRIVATE-CANARY/autogpt_desktop/postgres.py", co_firstlineno=1
    )
    try:
        trigger()
    except RuntimeError as exc:
        text = "".join(traceback.format_exception(exc))
    result = read_trace(driver, tmp_path, (TRACE_MARKER + text).replace("\n", newline))
    assert result == trace_result(
        [
            trace_exception(
                "os_error", [{"module": "autogpt_desktop.postgres", "line": 3}]
            ),
            trace_exception(
                "runtime_error",
                [
                    {
                        "module": "autogpt_desktop.postgres",
                        "line": 6 if chain == "cause" else 7,
                    }
                ],
                1,
            ),
        ]
    )
    assert "PRIVATE" not in json.dumps(result)


def test_startup_trace_python313_carets_unknown_frames_and_classes(driver, tmp_path):
    text = (
        TRACE_MARKER
        + TRACE_HEADER
        + TRACE_FRAME
        + "    await PRIVATE_FUNCTION(\n"
        + "        'PRIVATE-TOKEN-😀',\n"
        + "    )\n"
        + "    ^^^^^^^^^^^^^^^^^^^^^^\n"
        + '  File "C:/PRIVATE-CANARY/site-packages/unknown.py", line 99, in SECRET\n'
        + "    PRIVATE_LOCALS = 'PRIVATE-SECRET'\n"
        + "PRIVATE_CLASS: PRIVATE-VALUE\n"
        + "\nDuring handling of the above exception, another exception occurred:\n\n"
        + TRACE_HEADER
        + TRACE_FRAME
        + "subprocess.CalledProcessError: PRIVATE-ARGS\n"
    )
    assert read_trace(driver, tmp_path, text) == trace_result(
        [trace_exception("unknown", omitted=1), trace_exception("subprocess_error")]
    )


def startup_producer_log(monkeypatch, continuation):
    runtime = BUILD.parent / "runtime/autogpt_desktop"
    output = io.StringIO()
    logger = logging.getLogger("autogpt_desktop")
    monkeypatch.setattr(logging.root, "handlers", [])
    monkeypatch.setattr(logging.root, "level", logging.WARNING)
    monkeypatch.setattr(logger, "handlers", [])
    monkeypatch.setattr(logger, "propagate", True)
    monkeypatch.setattr(logger, "level", logging.NOTSET)
    monkeypatch.setattr(logger, "disabled", False)

    def definition(filename, name, namespace):
        path = runtime / filename
        tree = ast.parse(path.read_bytes())
        node = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        module = compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec")
        code = next(value for value in module.co_consts if isinstance(value, CodeType))
        return FunctionType(code, namespace)

    events_namespace = {
        "logging": logging,
        "sys": SimpleNamespace(stderr=output),
        "logger": logger,
        "_emit": lambda record: None,
    }
    definition("events.py", "configure_logging", events_namespace)()
    events = SimpleNamespace(error=definition("events.py", "error", events_namespace))
    run = definition(
        "supervisor.py", "run_until_stopped", {"logger": logger, "events": events}
    )

    def fail():
        try:
            inner = FileNotFoundError(
                "PRIVATE-INNER\nPRIVATE-CONTINUATION"
                if continuation == "inner_multiline"
                else "PRIVATE-INNER"
            )
            if continuation == "inner_note":
                inner.add_note("PRIVATE-NOTE")
            raise inner
        except FileNotFoundError as inner:
            outer = RuntimeError(
                "PRIVATE-OUTER\nPRIVATE-CONTINUATION"
                if continuation == "outer_multiline"
                else "PRIVATE-OUTER"
            )
            if continuation == "outer_note":
                outer.add_note("PRIVATE-NOTE")
            raise outer from inner

    fail.__code__ = fail.__code__.replace(
        co_filename="C:/PRIVATE/autogpt_desktop/postgres.py"
    )
    fake = SimpleNamespace(
        start=fail,
        stop_requested=SimpleNamespace(is_set=lambda: False),
        forget_password_reset=lambda: "",
    )
    assert run(fake) == 1
    return output.getvalue()


@pytest.mark.parametrize(
    "continuation",
    ["none", "inner_multiline", "inner_note", "outer_multiline", "outer_note"],
)
def test_startup_trace_actual_producer_chain_completeness(
    driver, tmp_path, monkeypatch, continuation
):
    text = startup_producer_log(monkeypatch, continuation)
    assert "[desktop] ERROR startup failed\nTraceback (most recent call last):" in text
    assert "RuntimeError: PRIVATE-OUTER" in text
    assert "[desktop] ERROR PRIVATE-OUTER" in text
    text += (
        "DESKTOP_SMOKE_RUNTIME_DIAGNOSTIC "
        + json.dumps(runtime_record())
        + "\nDESKTOP_SMOKE_DIAGNOSTIC "
        + json.dumps(diagnostic_record())
        + "\n"
    )
    result = read_trace(driver, tmp_path, text)
    if continuation == "none":
        assert [item["category"] for item in result["exceptions"]] == [
            "os_error",
            "runtime_error",
        ]
        assert any(
            frame["module"] == "autogpt_desktop.supervisor"
            for frame in result["exceptions"][1]["frames"]
        )
        assert not any(result[key] for key in ("rejected", "truncated", "unavailable"))
    else:
        assert result == trace_result(rejected=True)
    assert "PRIVATE" not in json.dumps(result)


@pytest.mark.parametrize(
    "boundary",
    [
        TRACE_MARKER.replace("startup failed", "PRIVATE-TRAILER"),
        TRACE_MARKER.replace("ERROR startup failed", "INFO PRIVATE-TRAILER"),
        "DESKTOP_SMOKE_DIAGNOSTIC {}\n",
        "DESKTOP_SMOKE_RUNTIME_DIAGNOSTIC {}\n",
        "== start 1 of 4: fresh install\n",
    ],
)
@pytest.mark.parametrize("later_chain", [False, True])
def test_startup_trace_boundary_cannot_hide_later_chain(
    driver, tmp_path, boundary, later_chain
):
    text = TRACE_MARKER + TRACE_BODY + boundary
    if later_chain:
        text += (
            "PRIVATE-CONTINUATION\n\n"
            "The above exception was the direct cause of the following exception:\n\n"
            + TRACE_BODY
        )
    result = read_trace(driver, tmp_path, text)
    assert result == (
        trace_result(rejected=True)
        if later_chain
        else trace_result([trace_exception()])
    )


@pytest.mark.parametrize(
    "continuation", ["PRIVATE-NOTE", "    PRIVATE-NOTE", "未知", "__proto__"]
)
def test_startup_trace_unrecognized_continuation_is_rejected(
    driver, tmp_path, continuation
):
    result = read_trace(
        driver, tmp_path, TRACE_MARKER + TRACE_BODY + continuation + "\n"
    )
    assert result == trace_result(rejected=True)


@pytest.mark.parametrize(
    "marker",
    [
        "[desktop] ERROR startup failed\n",
        TRACE_MARKER.replace("ERROR", "INFO"),
        TRACE_MARKER.replace("desktop", "other"),
        TRACE_MARKER.rstrip() + " PRIVATE\n",
        "PRIVATE " + TRACE_MARKER,
        '{"message":"' + TRACE_MARKER.rstrip() + '"}\n',
    ],
)
def test_startup_trace_requires_exact_logger_anchor(driver, tmp_path, marker):
    assert read_trace(driver, tmp_path, marker + TRACE_BODY) == trace_result(
        unavailable=True
    )


@pytest.mark.parametrize(
    "body",
    [
        "RuntimeError: PRIVATE\n",
        TRACE_HEADER + "RuntimeError: PRIVATE\n",
        TRACE_HEADER + TRACE_FRAME + '{"__proto__":{"category":"os_error"}}\n',
        TRACE_HEADER + TRACE_FRAME + "DESKTOP_SMOKE_DIAGNOSTIC {}\n",
        TRACE_HEADER
        + TRACE_FRAME.replace("210", "1000001")
        + "RuntimeError: PRIVATE\n",
        TRACE_HEADER + TRACE_FRAME.replace("210", "-1") + "RuntimeError: PRIVATE\n",
        TRACE_HEADER + TRACE_FRAME.replace("210", "２１０") + "RuntimeError: PRIVATE\n",
        TRACE_HEADER
        + TRACE_FRAME.replace("  File", "File")
        + "RuntimeError: PRIVATE\n",
        TRACE_BODY + TRACE_FRAME,
        TRACE_HEADER + TRACE_FRAME + TRACE_MARKER + TRACE_BODY,
        TRACE_HEADER + TRACE_FRAME + "\x1b[31mRuntimeError: PRIVATE\x1b[0m\n",
        TRACE_HEADER + TRACE_FRAME + "RuntimeError\r: PRIVATE\n",
    ],
)
def test_startup_trace_rejects_malformed_or_injected_structure(driver, tmp_path, body):
    result = read_trace(driver, tmp_path, TRACE_MARKER + body)
    assert result["rejected"] is True
    assert result["exceptions"] == []
    assert "PRIVATE" not in json.dumps(result)


@pytest.mark.parametrize(
    "path",
    [
        "C:/PRIVATE/autogpt_desktop/not_allowlisted.py",
        "C:/PRIVATE/other/supervisor.py",
        "C:/PRIVATE/autogpt_desktop/../supervisor.py",
        "https://PRIVATE/autogpt_desktop/supervisor.py",
        "autogpt_desktop/PRIVATE/supervisor.py",
    ],
)
def test_startup_trace_unknown_paths_are_only_counted(driver, tmp_path, path):
    text = TRACE_MARKER + TRACE_HEADER + f'  File "{path}", line 12, in PRIVATE\n'
    assert read_trace(driver, tmp_path, text + "ValueError: PRIVATE\n") == trace_result(
        [trace_exception("value_error", [], 1)]
    )


@pytest.mark.parametrize("suffix", [b"\xff\n", b"RuntimeError: cut", b"\xf0\x9f"])
def test_startup_trace_invalid_utf8_and_partial_file(driver, tmp_path, suffix):
    result = read_trace(
        driver, tmp_path, (TRACE_MARKER + TRACE_HEADER + TRACE_FRAME).encode() + suffix
    )
    assert result["exceptions"] == []
    assert result["rejected"] or result["truncated"]


@pytest.mark.parametrize("limit", ["bytes", "lines", "line", "frames", "exceptions"])
def test_startup_trace_resource_bounds(driver, tmp_path, limit):
    if limit == "bytes":
        text = ("x" * 1023 + "\n") * 1024 + TRACE_MARKER + TRACE_BODY
    elif limit == "lines":
        text = "\n" * 4096 + TRACE_MARKER + TRACE_BODY
    elif limit == "line":
        text = "x" * 1000000 + TRACE_MARKER + TRACE_BODY
    elif limit == "frames":
        text = (
            TRACE_MARKER + TRACE_HEADER + TRACE_FRAME * 33 + "RuntimeError: PRIVATE\n"
        )
    else:
        text = TRACE_MARKER + (
            "\nThe above exception was the direct cause of the following exception:\n\n"
        ).join([TRACE_BODY] * 9)
    result = read_trace(driver, tmp_path, text)
    assert result["truncated"] is True
    assert result["exceptions"] == []
    assert len(json.dumps(result)) < 200


def test_startup_trace_read_is_bounded_and_read_only(driver, tmp_path, monkeypatch):
    log = tmp_path / "lifecycle.log"
    log.write_bytes(b"synthetic")
    reads = []

    class Stream(io.BytesIO):
        def read(self, size=-1):
            reads.append(size)
            assert 0 < size <= 1024 * 1024 + 1
            return super().read(size)

    def opened(path, mode):
        assert path == log
        assert mode == "rb"
        return Stream((TRACE_MARKER + TRACE_BODY).encode())

    monkeypatch.setattr(Path, "open", opened)
    assert driver.startup_failure_trace(log) == trace_result([trace_exception()])
    assert reads == [1024 * 1024 + 1]


@pytest.mark.parametrize(
    "boundary", ["bytes", "lines", "line", "frames", "exceptions", "source_line"]
)
def test_startup_trace_exact_limits_are_accepted(driver, tmp_path, boundary):
    text = TRACE_MARKER + TRACE_BODY
    if boundary == "bytes":
        remaining = 1024 * 1024 - len(text.encode())
        text = (
            ("x" * 4094 + "\n") * (remaining // 4095) + "\n" * (remaining % 4095) + text
        )
    elif boundary == "lines":
        text = "\n" * (4096 - text.count("\n")) + text
    elif boundary == "line":
        text = (
            TRACE_MARKER
            + TRACE_HEADER
            + TRACE_FRAME
            + "RuntimeError: "
            + "x" * (4095 - len("RuntimeError: "))
            + "\n"
        )
    elif boundary == "frames":
        text = (
            TRACE_MARKER + TRACE_HEADER + TRACE_FRAME * 32 + "RuntimeError: PRIVATE\n"
        )
    elif boundary == "exceptions":
        text = (
            TRACE_MARKER
            + "\nThe above exception was the direct cause of the following exception:\n\n".join(
                [TRACE_BODY] * 8
            )
        )
    else:
        text = text.replace("line 210", "line 1000000")
    result = read_trace(driver, tmp_path, text)
    assert result["truncated"] is result["rejected"] is result["unavailable"] is False
    assert len(result["exceptions"]) == (8 if boundary == "exceptions" else 1)
    assert len(result["exceptions"][0]["frames"]) == (32 if boundary == "frames" else 1)
    assert "PRIVATE" not in json.dumps(result)


@pytest.mark.parametrize("missing", [True, False])
def test_startup_trace_unavailable_file_is_safe(driver, tmp_path, missing):
    log = tmp_path / "lifecycle.log"
    if not missing:
        log.mkdir()
    assert driver.startup_failure_trace(log) == trace_result(unavailable=True)


def test_startup_trace_reparse_path_is_not_opened(driver, tmp_path, monkeypatch):
    log = tmp_path / "lifecycle.log"
    monkeypatch.setattr(
        Path,
        "lstat",
        lambda _: SimpleNamespace(st_mode=0o100600, st_file_attributes=0x400),
    )

    def forbidden(*args, **kwargs):
        pytest.fail("reparse log must not be opened")

    monkeypatch.setattr(Path, "open", forbidden)
    assert driver.startup_failure_trace(log) == trace_result(unavailable=True)


@pytest.fixture
def trace_wrapper(driver, processes, artifact, tmp_path, monkeypatch, capsys):
    directory, desktop, _, _ = artifact
    work = tmp_path / "trace-work"
    monkeypatch.setattr(driver, "require_clean_host", lambda: None)
    monkeypatch.setenv("GITHUB_SHA", COMMIT)
    select = driver.select_artifact
    monkeypatch.setattr(
        driver, "select_artifact", lambda folder, sha: select(folder, sha, desktop)
    )
    monkeypatch.setattr(driver, "private_environment", lambda *args: {})
    monkeypatch.setattr(driver, "OwnedProcesses", lambda *args: processes.tracker)
    runtime = processes.install / "resources/runtime"
    monkeypatch.setattr(
        driver, "installed_runtime", lambda _: (runtime, runtime / "python.exe", {})
    )
    cleanup = processes.tracker.cleanup
    cleaned = []

    def clean():
        cleaned.append(True)
        return cleanup()

    monkeypatch.setattr(processes.tracker, "cleanup", clean)

    def run(failure, text=TRACE_MARKER + TRACE_BODY):
        def stage(name, command, timeout, private, env, tracker, stages):
            stages.append({"name": name, "exit_code": 0})
            if name != "lifecycle":
                return
            headings = "\n".join(
                f"== start {i} of 4: {title}"
                for i, title in enumerate(driver.STARTS, 1)
            )
            (private / "logs/lifecycle.log").write_text(
                text + headings + "\nOK in 1s\n", encoding="utf-8"
            )
            (processes.data / "config").mkdir(parents=True)
            (processes.data / "config/ports.json").write_text(
                '{"public":18473}', encoding="utf-8"
            )
            if failure is not None:
                stages[-1]["exit_code"] = 37
                processes.Process(
                    55, executable=processes.install / "runtime-worker.exe"
                )
                raise failure

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
        assert cleaned == [True]
        assert "PRIVATE" not in json.dumps(result) + capsys.readouterr().out
        assert (work / "private/logs/lifecycle.log").read_text("utf-8").startswith(text)
        if isinstance(failure, (RuntimeError, KeyboardInterrupt)):
            original_log = (work / "private/logs/driver.log").read_text("utf-8")
            assert type(failure).__name__ + ": PRIVATE-ORIGINAL" in original_log
            assert "PRIVATE-IO" not in original_log
            assert "PRIVATE-PARSE" not in original_log
        assert result["upload_allowlist"] == ["result.json"]
        return code, result

    return run


@pytest.mark.parametrize(
    "failure", ["lifecycle_exit", "lifecycle_timeout", "driver", "interrupt"]
)
@pytest.mark.parametrize("parser_failure", [None, "io", "parse"])
def test_startup_trace_failed_main_preserves_exit_and_full_cleanup(
    driver, processes, trace_wrapper, monkeypatch, failure, parser_failure
):
    original = (
        RuntimeError("PRIVATE-ORIGINAL")
        if failure == "driver"
        else (
            KeyboardInterrupt("PRIVATE-ORIGINAL")
            if failure == "interrupt"
            else driver.SmokeError(failure)
        )
    )
    if parser_failure == "io":
        opened = Path.open

        class Unreadable(io.BytesIO):
            def read(self, size=-1):
                raise OSError("PRIVATE-IO")

        def open_log(path, mode="r", *args, **kwargs):
            if path.name == "lifecycle.log" and mode == "rb":
                with opened(path, mode) as stream:
                    return Unreadable(stream.read())
            return opened(path, mode, *args, **kwargs)

        monkeypatch.setattr(Path, "open", open_log)
    elif parser_failure == "parse":

        def cannot_parse(*args):
            raise ValueError("PRIVATE-PARSE")

        monkeypatch.setattr(driver, "_parse_startup_failure_trace", cannot_parse)
    code, result = trace_wrapper(original)
    assert code == 1
    assert result["ok"] is False
    assert result["error"] == (
        failure
        if failure.startswith("lifecycle")
        else "driver_exception_private_logs_only"
    )
    assert result["stages"][-1]["exit_code"] == 37
    assert result["lifecycle"]["completed"] is True
    assert processes.killed == [55]
    assert result["cleanup"]["surviving_owned_pids"] == []
    assert result["cleanup"]["live_owned_during_cleanup"] == [
        {"pid": 55, "created": 101.0}
    ]
    assert result["startup_failure_trace"] == (
        trace_result(unavailable=True)
        if parser_failure
        else trace_result([trace_exception()])
    )


@pytest.mark.parametrize("continuation", ["inner_multiline", "inner_note"])
def test_startup_trace_rejected_chain_preserves_failed_main(
    driver, processes, trace_wrapper, monkeypatch, continuation
):
    text = startup_producer_log(monkeypatch, continuation)
    code, result = trace_wrapper(driver.SmokeError("lifecycle_exit"), text)
    assert code == 1 and result["ok"] is False
    assert result["error"] == "lifecycle_exit"
    assert result["stages"][-1]["exit_code"] == 37
    assert result["startup_failure_trace"] == trace_result(rejected=True)
    assert processes.killed == [55]
    assert result["cleanup"]["surviving_owned_pids"] == []
    assert result["cleanup"]["live_owned_during_cleanup"] == [
        {"pid": 55, "created": 101.0}
    ]


def test_startup_trace_success_never_reads_or_attaches(
    driver, trace_wrapper, monkeypatch
):
    def forbidden(*args):
        pytest.fail("success must not read the failure traceback")

    monkeypatch.setattr(driver, "startup_failure_trace", forbidden)
    try:
        raise ValueError("caller exception must not trigger failure diagnostics")
    except ValueError:
        code, result = trace_wrapper(None)
    assert code == 0
    assert result["ok"] is True
    assert "startup_failure_trace" not in result


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
    assert result["source_sha256"] == {
        name: driver.sha256(BUILD / name)
        for name in (
            "installed_smoke_windows.py",
            "smoke_test.py",
            "../runtime/tests/backend_contract.py",
            "../runtime/autogpt_desktop/winlinks.py",
        )
    }
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
