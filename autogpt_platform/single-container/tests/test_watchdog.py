from __future__ import annotations

import os
import queue
import signal
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parents[1]
WATCHDOG_PATH = ASSET_DIR / "watchdog.sh"

HARNESS = r"""
set -Eeuo pipefail
export AUTOGPT_RUNTIME_DIR="$1"
export AUTOGPT_READY_FILE="${AUTOGPT_RUNTIME_DIR}/ready"
export AUTOGPT_RUNTIME_ENV="${AUTOGPT_RUNTIME_DIR}/runtime.env"
export AUTOGPT_ASSET_DIR="$2"

source "$3"

HEALTHCHECK_COUNT=0
REAL_SLEEP="$(type -P sleep)"

wait_for_ready_file() {
  :
}

wait_for_initial_health() {
  printf 'harness-ready\n'
}

run_healthcheck() {
  local output="$1"
  ((HEALTHCHECK_COUNT += 1))
  printf 'healthcheck-%s\n' "${HEALTHCHECK_COUNT}"
  if [[ "${WATCHDOG_TEST_MODE}" == forced && "${HEALTHCHECK_COUNT}" -eq 1 ]]; then
    return 0
  fi
  printf 'expected test failure\n' >"${output}"
  return 1
}

stop_appliance() {
  printf 'harness-stopped\n'
  exit 0
}

if [[ "${WATCHDOG_TEST_MODE}" == forced ]]; then
  notify_check_timer_started() {
    local watchdog_pid="$$"
    printf 'timer-ready\n'
    (
      "${REAL_SLEEP}" 0.05
      kill -USR1 "${watchdog_pid}"
    ) &
  }
  sleep() {
    exec "${REAL_SLEEP}" "$@"
  }
elif [[ "${WATCHDOG_TEST_MODE}" == periodic ]]; then
  sleep() {
    :
  }
fi

main
"""


@unittest.skipUnless(
    os.name == "posix" and hasattr(signal, "SIGUSR1"),
    "watchdog signals require a POSIX host",
)
class WatchdogSchedulingTest(unittest.TestCase):
    def test_sigusr1_queued_before_timer_runs_without_delay(self) -> None:
        harness = r"""
set -Eeuo pipefail
export AUTOGPT_RUNTIME_DIR="$1"
export AUTOGPT_ASSET_DIR="$2"
source "$3"
trap queue_forced_check USR1
kill -USR1 "$$"
wait_for_next_check
printf 'trigger=%s\n' "${CHECK_TRIGGER}"
"""
        with tempfile.TemporaryDirectory() as runtime_dir:
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    harness,
                    "bash",
                    runtime_dir,
                    str(ASSET_DIR),
                    str(WATCHDOG_PATH),
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout, "trigger=forced\n")

    def test_periodic_timer_failure_is_not_treated_as_forced_check(self) -> None:
        harness = r"""
set -Eeuo pipefail
export AUTOGPT_RUNTIME_DIR="$1"
export AUTOGPT_ASSET_DIR="$2"
source "$3"
sleep() {
  return 7
}
wait_for_next_check
"""
        with tempfile.TemporaryDirectory() as runtime_dir:
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    harness,
                    "bash",
                    runtime_dir,
                    str(ASSET_DIR),
                    str(WATCHDOG_PATH),
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )

        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("watchdog check timer failed with status 7", result.stderr)

    def test_periodic_timer_runs_healthcheck_until_failure_limit(self) -> None:
        with tempfile.TemporaryDirectory() as runtime_dir:
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    HARNESS,
                    "bash",
                    runtime_dir,
                    str(ASSET_DIR),
                    str(WATCHDOG_PATH),
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
                env={**os.environ, "WATCHDOG_TEST_MODE": "periodic"},
            )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.count("healthcheck-"), 3)
        first = result.stderr.index("health failure 1/3 trigger=scheduled")
        second = result.stderr.index("health failure 2/3 trigger=scheduled")
        third = result.stderr.index("health failure 3/3 trigger=scheduled")
        self.assertLess(first, second)
        self.assertLess(second, third)
        self.assertIn("harness-stopped", result.stdout)

    def test_sigusr1_forces_ordered_checks_through_same_healthcheck(self) -> None:
        with tempfile.TemporaryDirectory() as runtime_dir:
            process = subprocess.Popen(
                [
                    "bash",
                    "-c",
                    HARNESS,
                    "bash",
                    runtime_dir,
                    str(ASSET_DIR),
                    str(WATCHDOG_PATH),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={**os.environ, "WATCHDOG_TEST_MODE": "forced"},
            )
            output: list[str] = []
            lines: queue.Queue[str] = queue.Queue()
            reader = threading.Thread(
                target=self._collect_lines,
                args=(process, output, lines),
                daemon=True,
            )
            reader.start()
            try:
                self._wait_for_line(process, output, lines, "harness-ready")
                self._wait_for_line(process, output, lines, "timer-ready")
                self._wait_for_line(
                    process,
                    output,
                    lines,
                    "health check passed trigger=forced",
                )
                for failure in range(1, 4):
                    self._wait_for_line(process, output, lines, "timer-ready")
                    self._wait_for_line(
                        process,
                        output,
                        lines,
                        f"health failure {failure}/3 trigger=forced",
                    )
                returncode = process.wait(timeout=5)
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)
                reader.join(timeout=5)
                if process.stdout is not None:
                    process.stdout.close()

        rendered = "".join(output)
        self.assertFalse(reader.is_alive(), rendered)
        self.assertEqual(returncode, 0, rendered)
        self.assertEqual(rendered.count("healthcheck-"), 4)
        self.assertIn("harness-stopped", rendered)

    def _wait_for_line(
        self,
        process: subprocess.Popen[str],
        output: list[str],
        lines: queue.Queue[str],
        expected: str,
    ) -> None:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                line = lines.get(timeout=deadline - time.monotonic())
            except queue.Empty:
                break
            if expected in line:
                return
        self.fail(
            f"watchdog did not emit {expected!r}; returncode={process.poll()}; "
            f"output={''.join(output)!r}"
        )

    def _collect_lines(
        self,
        process: subprocess.Popen[str],
        output: list[str],
        lines: queue.Queue[str],
    ) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            output.append(line)
            lines.put(line)


if __name__ == "__main__":
    unittest.main()
