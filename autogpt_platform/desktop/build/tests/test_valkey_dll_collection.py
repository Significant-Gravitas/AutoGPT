"""Exercise DLL collection without compiling or launching a Valkey executable."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

BUILD = Path(__file__).resolve().parents[1]


@pytest.fixture
def dll_scan(tmp_path):
    bash = (
        Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"
        if os.name == "nt"
        else Path(shutil.which("bash") or "/nonexistent")
    )
    if not bash.is_file():
        pytest.skip("Bash is unavailable")
    runtime = tmp_path / "runtime with spaces"
    output = tmp_path / "output with spaces"
    commands = tmp_path / "commands"
    for path in (runtime, output, commands):
        path.mkdir()
    objdump = commands / "objdump"
    objdump.write_text(
        '#!/usr/bin/env bash\n[[ "$1" == -p ]] || exit 2\n'
        '[[ "${2##*/}" != "${FAIL_IMPORT:-}" ]] || exit 37\n'
        'cat "$2"\n',
        encoding="utf-8",
        newline="\n",
    )
    objdump.chmod(0o755)

    def run(imports, libraries, *, fail=""):
        binary = output / "valkey-server.exe"
        binary.write_text(imports, encoding="utf-8")
        for name, contents in libraries.items():
            (runtime / name).write_text(contents, encoding="utf-8")
        env = os.environ.copy()
        env["FAIL_IMPORT"] = fail
        result = subprocess.run(
            [
                str(bash),
                "--noprofile",
                "--norc",
                "-c",
                (
                    'cd "$1" && export PATH="$PWD:$PATH"; '
                    'exec bash "$2" "$3" "$4" "$5"'
                ),
                "dll-scan",
                commands.as_posix(),
                (BUILD / "valkey-dlls.sh").as_posix(),
                binary.as_posix(),
                runtime.as_posix(),
                output.as_posix(),
            ],
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return result, output

    return run


def test_recursive_runtime_imports_include_transitive_dlls_once(dll_scan):
    result, output = dll_scan(
        "DLL Name: KERNEL32.dll\nDLL Name: MSYS-2.0.DLL\n",
        {
            "msys-2.0.dll": "DLL Name: msys-gcc_s-seh-1.dll\n",
            "msys-gcc_s-seh-1.dll": "DLL Name: msys-2.0.dll\n",
        },
    )
    assert result.returncode == 0, result.stderr
    assert sorted(path.name for path in output.iterdir()) == [
        "msys-2.0.dll",
        "msys-gcc_s-seh-1.dll",
        "valkey-server.exe",
    ]
    assert result.stdout.count("Bundled msys-2.0.dll") == 1


def test_missing_transitive_runtime_dependency_stops_collection(dll_scan):
    result, _ = dll_scan(
        "DLL Name: msys-2.0.dll\n",
        {"msys-2.0.dll": "DLL Name: msys-missing.dll\n"},
    )
    assert result.returncode != 0
    assert "Missing MSYS runtime dependency: msys-missing.dll" in result.stderr


def test_inspection_failure_is_not_hidden_by_a_pipeline(dll_scan):
    result, _ = dll_scan(
        "DLL Name: msys-2.0.dll\n",
        {"msys-2.0.dll": ""},
        fail="msys-2.0.dll",
    )
    assert result.returncode == 37


@pytest.mark.parametrize("imports", ["", "DLL Name: KERNEL32.dll\n"])
def test_missing_runtime_import_stops_collection(dll_scan, imports):
    result, _ = dll_scan(imports, {})
    assert result.returncode != 0
    assert "Executable has no MSYS runtime import" in result.stderr


@pytest.mark.parametrize("name", ["../msys-2.0.dll", "msys-unexpected.dll extra"])
def test_invalid_import_name_stops_collection(dll_scan, name):
    result, _ = dll_scan(f"DLL Name: {name}\n", {})
    assert result.returncode != 0
    assert "Invalid DLL import" in result.stderr
