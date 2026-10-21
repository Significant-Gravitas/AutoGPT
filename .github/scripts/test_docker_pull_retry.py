import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("docker-pull-with-retry.sh")
BASH = (
    str(Path("C:/Program Files/Git/bin/bash.exe"))
    if sys.platform == "win32"
    else shutil.which("bash")
)


class DockerPullRetryTests(unittest.TestCase):
    def run_pull(self, failures, args):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docker = root / "docker"
            docker.write_text(
                "#!/usr/bin/env bash\n"
                '[[ "$#" == 2 && "$1" == pull ]] || exit 90\n'
                'count=0; [[ ! -f "$COUNTER" ]] || read -r count < "$COUNTER"\n'
                'count=$((count + 1)); echo "$count" > "$COUNTER"\n'
                'printf "%s" "$2" > "$ARGUMENT"\n'
                "((count > FAILURES))\n",
                newline="\n",
            )
            sleep = root / "sleep"
            sleep.write_text(
                '#!/usr/bin/env bash\necho "$1" >> "$DELAYS"\n', newline="\n"
            )
            script = root / "pull.sh"
            script.write_text(SCRIPT.read_text(), newline="\n")
            docker.chmod(0o755)
            sleep.chmod(0o755)
            result = subprocess.run(
                [
                    BASH,
                    "-c",
                    'export PATH="$(cd "$STUB_BIN" && pwd):$PATH"; exec bash "$@"',
                    "bash",
                    str(script),
                    *args,
                ],
                env={
                    **os.environ,
                    "PATH": directory + os.pathsep + os.environ["PATH"],
                    "STUB_BIN": directory,
                    "COUNTER": str(root / "counter"),
                    "ARGUMENT": str(root / "argument"),
                    "DELAYS": str(root / "delays"),
                    "FAILURES": str(failures),
                },
                capture_output=True,
                text=True,
            )
            count = (
                int((root / "counter").read_text())
                if (root / "counter").exists()
                else 0
            )
            delays = (
                (root / "delays").read_text().splitlines()
                if (root / "delays").exists()
                else []
            )
            argument = (
                (root / "argument").read_text()
                if (root / "argument").exists()
                else None
            )
            return result, count, delays, argument

    def test_success_after_zero_one_or_two_failures(self):
        for failures in (0, 1, 2):
            with self.subTest(failures=failures):
                result, count, delays, argument = self.run_pull(failures, ["image:tag"])
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(count, failures + 1)
                self.assertEqual(delays, ["2", "4"][:failures])
                self.assertEqual(argument, "image:tag")

    def test_terminal_failure_is_not_reported_as_success(self):
        result, count, delays, _ = self.run_pull(3, ["image:tag"])
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(count, 3)
        self.assertEqual(delays, ["2", "4"])

    def test_invalid_arguments_never_call_docker(self):
        for args in ([], [""], ["one", "two"]):
            with self.subTest(args=args):
                result, count, _, _ = self.run_pull(0, args)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(count, 0)

    def test_image_remains_one_literal_argument(self):
        image = "image with spaces; no-command"
        result, _, _, argument = self.run_pull(0, [image])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(argument, image)
