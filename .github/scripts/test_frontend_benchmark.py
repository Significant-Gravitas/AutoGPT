import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark_frontend import ROOT, Benchmark, read_test_inventory


class FrontendBenchmarkTests(unittest.TestCase):
    def test_unavailable_candidate_is_recorded_and_other_measurements_continue(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch("benchmark_frontend.FRONTEND", ROOT):
                benchmark = Benchmark(output)
                self.assertEqual(
                    benchmark.run("missing", [str(output / "does-not-exist")]), 127
                )
                self.assertEqual(
                    benchmark.run("next", [sys.executable, "-c", "print('measured')"]),
                    0,
                )
            self.assertIn("Cannot start", (output / "missing.log").read_text())
            self.assertEqual((output / "next.log").read_text().strip(), "measured")

    def test_inventory_preserves_counts_and_rejects_incomplete_or_nonpassing_reports(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "junit.xml"
            for marker in ("<failure/>", "<error/>", "<skipped/>"):
                path.write_text(
                    f'<testsuites><testsuite><testcase classname="a" name="b">{marker}</testcase></testsuite></testsuites>'
                )
                with self.assertRaises(ValueError):
                    read_test_inventory(path)
            path.write_text(
                '<testsuites><testsuite><testcase classname="a" name="b"/><testcase classname="a" name="b"/></testsuite></testsuites>'
            )
            self.assertEqual(read_test_inventory(path)[("a", "b")], 2)
            path.write_text("<testsuites/>")
            with self.assertRaises(ValueError):
                read_test_inventory(path)

    def test_dependency_and_worker_benchmarks_refuse_local_execution(self):
        env = {
            key: value for key, value in os.environ.items() if key != "GITHUB_ACTIONS"
        }
        for mode in ("tools", "tests"):
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("benchmark_frontend.py")),
                    mode,
                    "--output",
                    ".",
                ],
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("hosted Linux CI", result.stderr)


if __name__ == "__main__":
    unittest.main()
