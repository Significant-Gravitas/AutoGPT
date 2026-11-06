"""The service host's contract with the backend, against the real one.

backend_contract.py does the checking, in an interpreter of its own: the
bundle's, since importing the backend needs all of its dependencies and
changes the interpreter it is imported into. Without an assembled bundle
this is skipped here; build/smoke_test.py runs the same script on every OS
and fails the build on it.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
DESKTOP = TESTS.parents[1]


def bundle() -> Path | None:
    configured = os.environ.get("AUTOGPT_DESKTOP_RUNTIME")
    root = Path(configured) if configured else DESKTOP / "build" / "runtime"
    return root if (root / "manifest.json").is_file() else None


def bundle_python(root: Path) -> Path:
    if sys.platform == "win32":
        return root / "python" / "python.exe"
    return root / "python" / "bin" / "python3"


def test_the_bundled_backend_still_fits_the_service_host():
    root = bundle()
    if root is None:
        pytest.skip(
            "no assembled bundle (build/runtime); the smoke test runs this check"
        )
    result = subprocess.run(
        [str(bundle_python(root)), "-B", str(TESTS / "backend_contract.py"), str(root)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdin=subprocess.DEVNULL,
        timeout=600,
    )
    findings = [
        line
        for line in result.stdout.splitlines()
        if line.startswith(("ok:", "BROKEN:"))
    ]
    broken = [line for line in findings if line.startswith("BROKEN:")]
    assert not broken, "\n".join(broken)
    assert result.returncode == 0, (result.stdout + result.stderr)[-4000:]
    assert len(findings) > 20, "the contract script checked next to nothing"
