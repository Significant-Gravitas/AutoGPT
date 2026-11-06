"""Native launch gate contracts without executing a compiled binary."""

import importlib
import json
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

BUILD = Path(__file__).resolve().parents[1]
WORKFLOW = BUILD.parents[2] / ".github/workflows/platform-desktop-build.yml"


def test_native_gate_is_required_before_assembly_and_retains_either_failure():
    workflow = WORKFLOW.read_text("utf-8")
    steps = re.split(r"^      - ", workflow, flags=re.MULTILINE)
    build = next(step for step in steps if "id: valkey\n" in step)
    gate = next(step for step in steps if "id: valkey_native\n" in step)
    assembly = next(
        step for step in steps if "run: python -B build/build_runtime.py" in step
    )
    upload = next(
        step for step in steps if "name: Retain failed Windows Valkey" in step
    )
    assert steps.index(build) < steps.index(gate) < steps.index(assembly)
    assert "shell: msys2 {0}" in build
    assert "shell: pwsh" in gate
    assert "if: runner.os == 'Windows'\n" in gate
    assert "steps.native_python.outputs.python-path" in gate
    assert "-B build/valkey_windows_check.py build/.cache/valkey-windows" in gate
    assert "exit $LASTEXITCODE" in gate
    setup = next(step for step in steps if "id: native_python\n" in step)
    assert "uses: actions/setup-python@" in setup
    assert steps.index(setup) < steps.index(build)
    assert "if:" not in assembly
    for step in (build, gate, assembly):
        assert "continue-on-error" not in step
        assert "|| true" not in step
    assert "failure() && runner.os == 'Windows'" in upload
    assert (
        "steps.valkey.outcome == 'failure' || steps.valkey_native.outcome == 'failure'"
        in upload
    )
    assert "include-hidden-files: true" in upload
    assert "valkey-windows/*.exe" in upload
    assert "valkey-windows/*.dll" in upload
    assert "valkey-windows-diagnostics/*.json" in upload
    assert "valkey-windows-diagnostics/*.txt" in upload


def test_msys_build_only_inspects_binaries_and_keeps_diagnostics_outside_bundle():
    script = (BUILD / "valkey-windows.sh").read_text("utf-8")
    assert '"${output}/valkey-server.exe" --version' not in script
    assert 'bash "${script_dir}/valkey-dlls.sh"' in script
    assert 'diagnostics="${output}-diagnostics"' in script
    assert '"${diagnostics}/toolchain.txt"' in script
    assert '"${diagnostics}/sha256.txt"' in script
    assert '"${diagnostics}/dependencies.txt"' in script
    assert '"${diagnostics}/$(basename "$binary").imports.txt"' in script


@pytest.fixture
def gate(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(BUILD))
    module = importlib.import_module("valkey_windows_check")
    monkeypatch.setattr(module, "WINDOWS", True)
    modes = []
    monkeypatch.setattr(
        module.ctypes,
        "windll",
        SimpleNamespace(
            kernel32=SimpleNamespace(
                SetErrorMode=lambda value: modes.append(value) or 0
            )
        ),
        raising=False,
    )
    monkeypatch.setattr(module.tempfile, "tempdir", str(tmp_path))
    output = tmp_path / "build-cache"
    output.mkdir()
    (output / "valkey-server.exe").write_bytes(b"not an executable")
    (output / "msys-2.0.dll").write_bytes(b"not a DLL")
    (output / "msys-gcc_s-seh-1.dll").write_bytes(b"another DLL")
    calls = []

    def run(command, **kwargs):
        checker = Path(kwargs["executable"])
        assert checker.name == "valkey-check-rdb.exe"
        assert checker.read_bytes() == (output / "valkey-server.exe").read_bytes()
        assert checker.parent != output
        assert not checker.is_relative_to(output)
        assert (checker.parent / "msys-2.0.dll").read_bytes() == b"not a DLL"
        assert (checker.parent / "msys-gcc_s-seh-1.dll").read_bytes() == b"another DLL"
        assert command == f'"{checker}" --version'
        assert kwargs["shell"] is False
        assert kwargs["timeout"] == 15
        assert kwargs["env"]["PATH"] == str(checker.parent)
        calls.append(checker)
        return subprocess.CompletedProcess(command, 0, version(module), "")

    monkeypatch.setattr(module.subprocess, "run", run)
    return module, output, calls, modes


def version(module):
    return (
        f"valkey-check-rdb v={module.VALKEY_VERSION} "
        "sha=00000000:0 malloc=libc bits=64 build=3842ead13eef2145\n"
    )


def invoke(module, output):
    status = module.main([str(output)])
    report = json.loads(
        output.with_name(output.name + "-diagnostics")
        .joinpath("native-launch.json")
        .read_text("utf-8")
    )
    assert report["ok"] is (status == 0)
    return status, report


def test_identical_copy_validates_pinned_version_and_leaves_cache_unchanged(gate):
    module, output, calls, modes = gate
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    status, report = invoke(module, output)
    assert status == 0
    assert report["expected_version"] == module.VALKEY_VERSION
    assert report["server_sha256"] == report["checker_sha256"]
    assert report["windows_status"] == "0x00000000"
    assert report["stdout"] == version(module)
    assert modes == [0x0001 | 0x0002 | 0x8000, 0]
    assert len(calls) == 1 and not calls[0].parent.exists()
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


@pytest.mark.parametrize(
    "profile", ["O'Brien", "John Smith", "Jürgen", "O''Brien", "O'Brien's"]
)
def test_probe_executable_is_explicitly_quoted_for_profile_paths(
    gate, monkeypatch, tmp_path, profile
):
    module, output, calls, _ = gate
    temporary = tmp_path / profile
    temporary.mkdir()
    monkeypatch.setattr(module.tempfile, "tempdir", str(temporary))
    assert invoke(module, output)[0] == 0
    assert profile in str(calls[0])


@pytest.mark.parametrize(
    "stdout",
    [
        "",
        "usage: valkey-check-rdb <file>\n",
        "valkey-check-rdb v=8.1.1 sha=00000000:0 malloc=libc bits=64 build=3842ead13eef2145\n",
        "valkey-check-rdb v=8.1.100 sha=00000000:0 malloc=libc bits=64 build=3842ead13eef2145\n",
        "valkey-server v=8.1.10 sha=00000000:0 malloc=libc bits=64 build=3842ead13eef2145\n",
        "valkey-check-rdb v=8.1.10\n",
    ],
)
def test_zero_exit_without_exact_version_output_fails(gate, monkeypatch, stdout):
    module, output, _, _ = gate
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, stdout, ""),
    )
    status, report = invoke(module, output)
    assert status == 1
    assert report["stdout"] == stdout
    assert "version" in report["error"]
    assert not list(output.parent.glob("valkey-native-*"))


@pytest.mark.parametrize("returncode", [1, 127, 3221225781, -1073741515])
def test_nonzero_exit_preserves_raw_status_even_with_matching_version(
    gate, monkeypatch, returncode
):
    module, output, _, _ = gate
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], returncode, version(module), "loader failure"
        ),
    )
    status, report = invoke(module, output)
    assert status == 1
    assert report["returncode"] == returncode
    assert report["windows_status"] == f"0x{returncode & 0xffffffff:08X}"
    assert report["stderr"] == "loader failure"


@pytest.mark.parametrize("name", ["valkey-server.exe", "msys-2.0.dll"])
@pytest.mark.parametrize("empty", [False, True])
def test_missing_or_empty_required_binary_fails_before_launch(gate, name, empty):
    module, output, calls, _ = gate
    if empty:
        (output / name).write_bytes(b"")
    else:
        (output / name).unlink()
    status, report = invoke(module, output)
    assert status == 1 and not calls
    assert name in report["error"]


def test_zero_exit_with_stderr_is_not_a_success(gate, monkeypatch):
    module, output, _, _ = gate
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, version(module), "loader error"
        ),
    )
    status, report = invoke(module, output)
    assert status == 1
    assert report["stderr"] == "loader error"


def test_changed_checker_bytes_fail_before_launch(gate, monkeypatch):
    module, output, calls, _ = gate
    monkeypatch.setattr(
        module.shutil, "copy2", lambda source, target: target.write_bytes(b"changed")
    )
    status, report = invoke(module, output)
    assert status == 1 and not calls
    assert report["checker_sha256"] != report["server_sha256"]
    assert not list(output.parent.glob("valkey-native-*"))


def test_loader_error_is_retained_and_probe_is_cleaned(gate, monkeypatch):
    module, output, _, modes = gate
    error = OSError("DLL loader failure")
    error.winerror = 126

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(module.subprocess, "run", fail)
    status, report = invoke(module, output)
    assert status == 1
    assert report["winerror"] == 126
    assert report["error"] == str(error)
    assert modes[-1] == 0
    assert not list(output.parent.glob("valkey-native-*"))


def test_timeout_is_failure_with_partial_output_and_cleanup(gate, monkeypatch):
    module, output, _, modes = gate

    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(
            command, 15, output=b"partial\xff", stderr=b"loader stalled"
        )

    monkeypatch.setattr(module.subprocess, "run", timeout)
    status, report = invoke(module, output)
    assert status == 1
    assert report["timeout_seconds"] == 15
    assert report["stdout"] == "partial\ufffd"
    assert report["stderr"] == "loader stalled"
    assert modes[-1] == 0
    assert not list(output.parent.glob("valkey-native-*"))


def test_non_native_python_fails_without_launch(gate, monkeypatch):
    module, output, calls, _ = gate
    monkeypatch.setattr(module, "WINDOWS", False)
    status, report = invoke(module, output)
    assert status == 1 and not calls
    assert "native Windows Python" in report["error"]
