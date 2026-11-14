import subprocess
import sys


def test_recording_tools_imports_in_fresh_process() -> None:
    # Guards against an import cycle, not import speed: a cold import of the
    # backend takes 20+ seconds on a loaded machine, so the bound is generous.
    result = subprocess.run(
        [sys.executable, "-c", "import backend.copilot.sdk.recording_tools"],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stderr
