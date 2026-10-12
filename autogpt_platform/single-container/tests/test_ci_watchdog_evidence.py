import re
import subprocess
import unittest

from test_ci_smoke_stop import SMOKE, bash_executable


class WatchdogEvidenceTests(unittest.TestCase):
    def run_case(
        self,
        *,
        already_seen=0,
        previous_count=0,
        signal_adds_evidence=1,
        signal_status=0,
        observed_ordinal=2,
    ):
        source = SMOKE.read_text(encoding="utf-8")
        functions = []
        for name in (
            "count_container_log_evidence",
            "wait_for_new_container_log_evidence",
            "advance_watchdog_failure",
        ):
            match = re.search(
                rf"^{name}\(\) \{{.*?^\}}", source, re.MULTILINE | re.DOTALL
            )
            self.assertIsNotNone(match, f"missing {name}")
            functions.append(match.group())
        script = (
            "\n".join(functions)
            + f"""
set -Eeuo pipefail
TIMEOUT_SECONDS=2
CONTAINER_NAME=test
COUNT={already_seen}
docker() {{
  if [[ "$1" == logs ]]; then
    for ((i=0; i<COUNT; i++)); do
      printf '[single-container] watchdog health failure {observed_ordinal}/3 trigger=scheduled\\n'
    done
    return 0
  fi
  printf 'signal\\n'
  COUNT=$((COUNT + {signal_adds_evidence}))
  return {signal_status}
}}
advance_watchdog_failure 123 2 {previous_count}
"""
        )
        result = subprocess.run(
            [bash_executable(), "--noprofile", "--norc", "-s"],
            input=script.encode("utf-8"),
            capture_output=True,
            timeout=5,
            check=False,
        )
        return subprocess.CompletedProcess(
            result.args,
            result.returncode,
            result.stdout.decode("utf-8"),
            result.stderr.decode("utf-8"),
        )

    def test_scheduled_failure_before_signal_needs_no_extra_signal(self):
        result = self.run_case(already_seen=1)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("signal", result.stdout)

    def test_forced_check_still_requires_new_failure_evidence(self):
        result = self.run_case()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("signal", result.stdout)

    def test_restart_racing_with_signal_accepts_observed_failure(self):
        result = self.run_case(signal_status=1)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_failed_signal_without_evidence_fails(self):
        result = self.run_case(signal_adds_evidence=0, signal_status=1)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("could not advance watchdog failure", result.stderr)

    def test_successful_signal_without_evidence_fails(self):
        result = self.run_case(signal_adds_evidence=0)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("did not report new watchdog evidence", result.stderr)

    def test_old_failure_evidence_does_not_satisfy_new_check(self):
        result = self.run_case(already_seen=1, previous_count=1)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("signal", result.stdout)

    def test_different_failure_ordinal_does_not_satisfy_check(self):
        result = self.run_case(observed_ordinal=3)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("did not report new watchdog evidence", result.stderr)


if __name__ == "__main__":
    unittest.main()
