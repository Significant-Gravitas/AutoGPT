import subprocess
import sys


def test_recording_tools_imports_in_fresh_process() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import backend.copilot.sdk.recording_tools"],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
