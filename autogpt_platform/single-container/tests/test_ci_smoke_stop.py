import re
import subprocess
import unittest
from pathlib import Path


SMOKE = (
    Path(__file__).resolve().parents[3]
    / ".github/scripts/platform-single-container-smoke.sh"
)


class SmokeStopTests(unittest.TestCase):
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
                    ["bash", "--noprofile", "--norc", "-s"],
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
