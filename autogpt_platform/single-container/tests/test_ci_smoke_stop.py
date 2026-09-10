import re
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


SMOKE = (
    Path(__file__).resolve().parents[3]
    / ".github/scripts/platform-single-container-smoke.sh"
)


def bash_executable() -> str:
    if sys.platform != "win32":
        return "/bin/bash"
    executable = shutil.which("bash")
    if executable is None:
        raise RuntimeError("Git Bash must be installed and on PATH for this test")
    return executable


class SmokeStopTests(unittest.TestCase):
    def test_uses_fixed_bash_path_on_posix(self):
        with patch("sys.platform", "linux"), patch("shutil.which") as which:
            self.assertEqual(bash_executable(), "/bin/bash")
            which.assert_not_called()

    def test_resolves_bash_executable_on_windows(self):
        selected = "C:/Program Files/Git/bin/bash.exe"
        with patch("sys.platform", "win32"), patch(
            "shutil.which", return_value=selected
        ) as which:
            self.assertEqual(bash_executable(), selected)
            which.assert_called_once_with("bash")

    def test_missing_windows_bash_fails_instead_of_skipping(self):
        with patch("sys.platform", "win32"), patch("shutil.which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "Git Bash"):
                bash_executable()

    def test_invalid_or_empty_finish_time_fails_with_diagnostic(self):
        function = re.search(
            r"^assert_clean_stop\(\) \{.*?^\}",
            SMOKE.read_text(encoding="utf-8"),
            re.MULTILINE | re.DOTALL,
        ).group()
        for date_status in (0, 1):
            with self.subTest(date_status=date_status):
                script = (
                    "set -Eeuo pipefail\n"
                    "STOCK_DOCKER_STOP_TIMEOUT=10\n"
                    "CONTAINER_NAME=test-container\n"
                    "docker() { printf '%s' invalid-timestamp; }\n"
                    f"date() {{ return {date_status}; }}\n"
                    "awk() { echo UNEXPECTED_AWK >&2; return 1; }\n"
                    f"{function}\n"
                    "assert_clean_stop test-stop\n"
                )
                result = subprocess.run(
                    [bash_executable(), "--noprofile", "--norc", "-s"],
                    input=script.encode("utf-8"),
                    capture_output=True,
                    check=False,
                )
                stderr = result.stderr.decode("utf-8")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("invalid container finish timestamp", stderr)
                self.assertNotIn("UNEXPECTED_AWK", stderr)


if __name__ == "__main__":
    unittest.main()
