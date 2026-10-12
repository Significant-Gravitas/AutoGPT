import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import flake_gate
from flake_gate import (
    FAILED,
    PASSED,
    SKIPPED,
    BackendSuite,
    FrontendSuite,
    Outcome,
    Pressure,
    Run,
    annotation_commands,
    changed_files,
    classify_changes,
    render,
    run_gate,
)
from flake_gate_pytest import shuffled


def _run(index, seed, outcomes, *, duration=1.0, stopped_after=None, exit_code=0):
    return Run(
        index=index,
        seed=seed,
        load=8,
        duration=duration,
        exit_code=exit_code,
        stopped_after=stopped_after,
        reported=True,
        outcomes=outcomes,
        log=Path(f"run-{index}.log"),
    )


class DetectTests(unittest.TestCase):
    def test_picks_the_test_files_each_suite_runs(self):
        changes = classify_changes(
            [
                "autogpt_platform/backend/backend/data/credit_test.py",
                "autogpt_platform/backend/test/blocks/test_youtube.py",
                "autogpt_platform/backend/scripts/run_tests_test.py",
                "autogpt_platform/backend/backend/data/credit.py",
                "autogpt_platform/backend/backend/data/test_fixtures.json",
                "autogpt_platform/autogpt_libs/autogpt_libs/auth/jwt_utils_test.py",
                "autogpt_platform/frontend/src/lib/keyboard.test.ts",
                "autogpt_platform/frontend/src/app/x/__tests__/main.test.tsx",
                "autogpt_platform/frontend/scripts/generate.test.ts",
                "autogpt_platform/frontend/src/playwright/login.spec.ts",
                "autogpt_platform/frontend/src/lib/keyboard.ts",
                ".github/scripts/test_validate_junit.py",
            ]
        )
        self.assertEqual(
            changes.backend,
            [
                "autogpt_platform/backend/backend/data/credit_test.py",
                "autogpt_platform/backend/test/blocks/test_youtube.py",
                "autogpt_platform/backend/scripts/run_tests_test.py",
            ],
        )
        self.assertEqual(
            changes.frontend,
            [
                "autogpt_platform/frontend/src/lib/keyboard.test.ts",
                "autogpt_platform/frontend/src/app/x/__tests__/main.test.tsx",
                "autogpt_platform/frontend/scripts/generate.test.ts",
            ],
        )
        self.assertFalse(changes.gate)
        self.assertTrue(classify_changes([".github/scripts/flake_gate.py"]).gate)

    def test_lists_added_modified_and_renamed_files_but_not_deleted_ones(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)

            def git(*args):
                subprocess.run(
                    ["git", "-c", "commit.gpgsign=false", *args],
                    cwd=repo,
                    check=True,
                    capture_output=True,
                )

            git("init", "-q")
            git("config", "user.email", "gate@example.com")
            git("config", "user.name", "gate")
            for name in ("kept.py", "modified.py", "deleted.py", "renamed.py"):
                (repo / name).write_text(f"# {name}\n" * 20)
            git("add", ".")
            git("commit", "-qm", "base")
            (repo / "modified.py").write_text("changed\n")
            (repo / "deleted.py").unlink()
            (repo / "renamed.py").rename(repo / "renamed_to.py")
            (repo / "added dir").mkdir()
            (repo / "added dir" / "new file.py").write_text("new\n")
            git("add", "-A")
            git("commit", "-qm", "change")

            with mock.patch.object(flake_gate, "REPO_ROOT", repo):
                self.assertEqual(
                    changed_files("HEAD^1", "HEAD"),
                    ["added dir/new file.py", "modified.py", "renamed_to.py"],
                )


class ShuffleTests(unittest.TestCase):
    NODEIDS = [
        "a_test.py::test_1",
        "a_test.py::test_2[x::y]",
        "a_test.py::test_2[z]",
        "a_test.py::TestGroup::test_3",
        "a_test.py::TestGroup::test_4",
        "b_test.py::test_5",
        "b_test.py::test_6",
    ]

    def order(self, seed):
        items = [SimpleNamespace(nodeid=n) for n in self.NODEIDS]
        return [item.nodeid for item in shuffled(items, seed)]

    def test_a_seed_gives_the_same_order_every_time(self):
        self.assertEqual(self.order(5), self.order(5))
        orders = {tuple(self.order(seed)) for seed in range(20)}
        self.assertGreater(len(orders), 1)

    def test_keeps_files_and_classes_together(self):
        for seed in range(50):
            order = self.order(seed)
            self.assertCountEqual(order, self.NODEIDS)
            files = [n.split("::")[0] for n in order]
            blocks = [f for i, f in enumerate(files) if i == 0 or files[i - 1] != f]
            self.assertEqual(sorted(blocks), ["a_test.py", "b_test.py"])
            group = [n for n in order if "::TestGroup::" in n]
            start = order.index(group[0])
            self.assertEqual(order[start : start + len(group)], group)


class BackendReportTests(unittest.TestCase):
    def parse(self, *records, tail=""):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "run.jsonl"
            report.write_text("".join(json.dumps(r) + "\n" for r in records) + tail)
            return BackendSuite().parse(report)

    def test_reads_every_phase_and_names_a_test_cut_off_by_a_stop(self):
        outcomes = self.parse(
            {"nodeid": "a_test.py::ok", "phase": "start", "line": 3},
            {"nodeid": "a_test.py::ok", "phase": "call", "outcome": "passed"},
            {"nodeid": "a_test.py::teardown_fails", "phase": "start", "line": 7},
            {
                "nodeid": "a_test.py::teardown_fails",
                "phase": "call",
                "outcome": "passed",
            },
            {
                "nodeid": "a_test.py::teardown_fails",
                "phase": "teardown",
                "outcome": "failed",
                "message": "E   boom",
            },
            {"nodeid": "a_test.py::skips", "phase": "start", "line": 9},
            {"nodeid": "a_test.py::skips", "phase": "setup", "outcome": "skipped"},
            {
                "nodeid": "b_test.py",
                "phase": "collect",
                "outcome": "failed",
                "message": "E   ImportError",
            },
            {"nodeid": "a_test.py::hangs", "phase": "start", "line": 12},
            tail='{"nodeid": "a_test.py::half',
        )
        self.assertEqual(outcomes["a_test.py::ok"], Outcome(PASSED, line=3))
        self.assertEqual(outcomes["a_test.py::teardown_fails"].status, FAILED)
        self.assertEqual(outcomes["a_test.py::teardown_fails"].message, "E   boom")
        self.assertEqual(outcomes["a_test.py::skips"].status, SKIPPED)
        self.assertEqual(outcomes["b_test.py"].status, FAILED)
        hung = outcomes["a_test.py::hangs"]
        self.assertEqual((hung.status, hung.line), (FAILED, 12))
        self.assertIn("never finished", hung.message)


class FrontendReportTests(unittest.TestCase):
    def test_reads_tests_and_files_that_failed_to_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            report = directory / "report.json"
            report.write_text(
                json.dumps(
                    {
                        "success": False,
                        "testResults": [
                            {
                                "name": str(directory / "src/a.test.tsx"),
                                "status": "failed",
                                "message": "",
                                "assertionResults": [
                                    {
                                        "ancestorTitles": ["page"],
                                        "title": "shows it",
                                        "status": "passed",
                                        "failureMessages": [],
                                    },
                                    {
                                        "ancestorTitles": ["page", "menu"],
                                        "title": "closes on Escape",
                                        "status": "failed",
                                        "failureMessages": [
                                            "AssertionError: expected <span> to be null\n"
                                            "    at src/a.test.tsx:9:3"
                                        ],
                                    },
                                    {
                                        "ancestorTitles": [],
                                        "title": "later",
                                        "status": "skipped",
                                        "failureMessages": [],
                                    },
                                ],
                            },
                            {
                                "name": str(directory / "src/b.test.ts"),
                                "status": "failed",
                                "message": 'Failed to resolve import "./gone"',
                                "assertionResults": [],
                            },
                        ],
                    }
                )
            )
            outcomes = FrontendSuite(directory=directory).parse(report)
        self.assertEqual(outcomes["src/a.test.tsx > page > shows it"].status, PASSED)
        failed = outcomes["src/a.test.tsx > page > menu > closes on Escape"]
        self.assertEqual(failed.status, FAILED)
        self.assertIn("expected <span> to be null", failed.message)
        self.assertEqual(outcomes["src/a.test.tsx > later"].status, SKIPPED)
        self.assertNotIn("src/a.test.tsx", outcomes)
        self.assertEqual(outcomes["src/b.test.ts"].status, FAILED)


class GateTests(unittest.TestCase):
    """The run loop and its verdicts, with the test runs faked."""

    FILES = ["a_test.py"]

    def gate(self, runs_by_index, *, budget=300, clock_values=None, runs=3):
        """Run the gate with ``execute`` answering from ``runs_by_index``."""
        calls = []

        def execute(suite, files, *, index, seed, pressure, timeout, out_dir):
            load = pressure.load if pressure else 0
            calls.append(
                SimpleNamespace(index=index, seed=seed, load=load, timeout=timeout)
            )
            run = runs_by_index[index]
            run.seed, run.load = seed, load
            return run

        values = iter(clock_values or [0.0] * 100)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            flake_gate, "execute", execute
        ):
            result = run_gate(
                BackendSuite(directory=Path(tmp)),
                self.FILES,
                runs=runs,
                budget=budget,
                pressure=Pressure(load=8),
                seed=100,
                out_dir=Path(tmp),
                clock=lambda: next(values),
                log=lambda line: None,
            )
        return result, calls

    def test_a_test_that_fails_only_under_load_is_flaky(self):
        ok = {"a_test.py::t": Outcome(PASSED), "a_test.py": Outcome(PASSED)}
        slow = {
            "a_test.py::t": Outcome(FAILED, "E   assert 1.62 < 1.5", line=4),
            "a_test.py": Outcome(PASSED),
        }
        result, calls = self.gate(
            {
                0: _run(0, None, ok),
                1: _run(1, 0, slow),
                2: _run(2, 0, ok),
                3: _run(3, 0, slow),
            }
        )
        self.assertEqual([c.seed for c in calls], [None, 100, 101, 102])
        self.assertEqual([c.load for c in calls], [0, 8, 8, 8])
        (verdict,) = result.of_kind("flaky")
        self.assertEqual(verdict.test, "a_test.py::t")
        self.assertEqual([r.seed for r, _ in verdict.failures], [100, 102])

        report = render(result)
        self.assertIn("❌ 1 flaky test", report)
        self.assertIn("2 of 4 runs (passed without load)", report)
        self.assertIn("E   assert 1.62 < 1.5", report)
        self.assertIn(
            "PYTHONHASHSEED=100 python3 ../../.github/scripts/flake_gate.py stress "
            "--load 8 -- poetry run test -p flake_gate_pytest "
            "--flake-gate-seed=100 a_test.py",
            report,
        )
        (annotation,) = annotation_commands(result)
        self.assertTrue(annotation.startswith("::error file="), annotation)
        self.assertIn("a_test.py,line=4,title=Flaky under CPU load::", annotation)

    def test_a_test_that_failed_only_without_load_is_flaky_too(self):
        ok = {"a_test.py::t": Outcome(PASSED), "a_test.py": Outcome(PASSED)}
        failed = {
            "a_test.py::t": Outcome(FAILED, "E   boom"),
            "a_test.py": Outcome(PASSED),
        }
        result, _ = self.gate({0: _run(0, None, failed), 1: _run(1, 0, ok)}, runs=1)
        self.assertEqual([v.test for v in result.of_kind("flaky")], ["a_test.py::t"])
        self.assertIn("\npoetry run test a_test.py\n", render(result))

    def test_a_test_that_fails_every_run_is_listed_but_not_counted(self):
        broken = {
            "a_test.py::t": Outcome(FAILED, "E   needs ClamAV"),
            "a_test.py": Outcome(PASSED),
        }
        result, _ = self.gate({i: _run(i, None, dict(broken)) for i in range(4)})
        self.assertEqual(result.of_kind("flaky"), [])
        self.assertEqual([v.test for v in result.of_kind("broken")], ["a_test.py::t"])
        report = render(result)
        self.assertIn("✅ no flaky tests", report)
        self.assertIn("**Not counted**", report)
        self.assertEqual(annotation_commands(result), [])

    def test_stops_when_the_next_run_would_not_fit_the_budget(self):
        ok = {"a_test.py::t": Outcome(PASSED)}
        # The clock reads 0 at the start, 200 after the run without load and
        # 250 after the first loaded run. That run without load took 40 s,
        # so a loaded one is first guessed at 80 s and fits the 100 s left;
        # the first took 60 s, which doesn't fit the 50 s left after it.
        result, calls = self.gate(
            {0: _run(0, None, ok, duration=40), 1: _run(1, 0, ok, duration=60)},
            budget=300,
            clock_values=[0, 200, 250, 260],
        )
        self.assertEqual([c.index for c in calls], [0, 1])
        self.assertEqual(calls[1].timeout, 100)
        self.assertIn("1 of 3 loaded runs fit the budget", result.note)

    def test_a_loaded_run_stopped_as_stuck_counts_against_the_test(self):
        ok = {"a_test.py::t": Outcome(PASSED), "a_test.py": Outcome(PASSED)}
        hung = {
            "a_test.py::t": Outcome(FAILED, "The test never finished"),
            "a_test.py": Outcome(PASSED),
        }
        # 10 s without load: stuck means 120 s, the floor, not 10 x 10 s.
        result, calls = self.gate(
            {
                0: _run(0, None, ok, duration=10),
                1: _run(1, 0, hung, duration=120, stopped_after=120),
            },
            budget=1000,
            runs=1,
        )
        self.assertEqual([c.timeout for c in calls], [1000, 120])
        self.assertEqual([v.test for v in result.of_kind("flaky")], ["a_test.py::t"])

    def test_a_loaded_run_cut_off_by_the_budget_is_not_counted(self):
        ok = {"a_test.py::t": Outcome(PASSED), "a_test.py": Outcome(PASSED)}
        cut = {
            "a_test.py::t": Outcome(FAILED, "The test never finished"),
            "a_test.py": Outcome(PASSED),
        }
        # 50 s are left for the second loaded run, less than the 120 s that
        # would mark it stuck, so stopping it there says nothing about it.
        result, calls = self.gate(
            {
                0: _run(0, None, ok, duration=10),
                1: _run(1, 0, ok, duration=20),
                2: _run(2, 0, cut, duration=50, stopped_after=50),
            },
            budget=1000,
            clock_values=[0, 10, 950, 1000],
        )
        self.assertEqual([c.timeout for c in calls], [1000, 120, 50])
        self.assertEqual([r.index for r in result.runs], [0, 1])
        self.assertEqual(result.of_kind("flaky"), [])
        self.assertIn("ran out of budget and was not counted", result.note)

    def test_tests_too_slow_for_the_budget_are_not_rerun(self):
        result, calls = self.gate({0: _run(0, None, {}, stopped_after=300)})
        self.assertEqual(len(calls), 1)
        self.assertIn("took more than the 300 s budget", result.note)
        self.assertFalse(result.error)

    def test_a_gate_that_cannot_run_the_tests_says_so(self):
        run = _run(0, None, {}, exit_code=4)
        run.reported = False
        result, _ = self.gate({0: run})
        self.assertTrue(result.error)
        self.assertIn("could not run the tests", render(result))


@unittest.skipUnless(find_spec("pytest"), "the plugin needs pytest")
class EndToEndTests(unittest.TestCase):
    """The gate driving real pytest runs through the plugin."""

    def test_flags_a_flaky_test_and_reproduces_its_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite_dir = Path(tmp)
            (suite_dir / "pytest.ini").write_text("[pytest]\n")
            (suite_dir / "sample_test.py").write_text(
                textwrap.dedent(
                    """
                    import os
                    import pathlib
                    import random

                    COUNTER = pathlib.Path(os.environ["FLAKE_GATE_SELF_TEST_COUNTER"])

                    def test_stable():
                        pass

                    def test_fails_every_other_run():
                        runs = int(COUNTER.read_text()) if COUNTER.exists() else 0
                        COUNTER.write_text(str(runs + 1))
                        assert runs % 2 == 0, f"run {runs}"

                    def test_always_fails():
                        assert False, "broken"

                    class TestGroup:
                        def test_random(self):
                            assert 0 <= random.random() < 1

                        def test_other(self):
                            pass
                    """
                )
            )
            env = {
                k: v
                for k, v in os.environ.items()
                if not k.startswith(("COV_CORE_", "COVERAGE_", "PYTEST_"))
            }
            env["FLAKE_GATE_SELF_TEST_COUNTER"] = str(suite_dir / "counter")
            # Only this plugin: a backend venv's own plugins clash without
            # the backend's pytest config.
            env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
            suite = BackendSuite(
                prefix=[sys.executable, "-m", "pytest"], directory=suite_dir, options=[]
            )
            with mock.patch.dict(os.environ, env, clear=True):
                result = run_gate(
                    suite,
                    ["sample_test.py"],
                    runs=3,
                    budget=600,
                    pressure=Pressure(load=0),
                    seed=41,
                    out_dir=suite_dir / "out",
                    log=lambda line: None,
                )

        kinds = {v.test: v.kind for v in result.verdicts.values()}
        self.assertEqual(
            kinds,
            {
                "sample_test.py": "stable",
                "sample_test.py::TestGroup::test_other": "stable",
                "sample_test.py::TestGroup::test_random": "stable",
                "sample_test.py::test_always_fails": "broken",
                "sample_test.py::test_fails_every_other_run": "flaky",
                "sample_test.py::test_stable": "stable",
            },
        )
        (flaky,) = result.of_kind("flaky")
        self.assertEqual([r.seed for r, _ in flaky.failures], [41, 43])
        self.assertEqual(flaky.line, 11)
        self.assertIn("run 1", flaky.failures[0][1].message)


class DetectCommandTests(unittest.TestCase):
    def detect(self, paths):
        with tempfile.TemporaryDirectory() as tmp:
            output, summary = Path(tmp) / "output", Path(tmp) / "summary"
            env = {"GITHUB_OUTPUT": str(output), "GITHUB_STEP_SUMMARY": str(summary)}
            with mock.patch.dict(os.environ, env), mock.patch.object(
                flake_gate, "changed_files", return_value=paths
            ), mock.patch("builtins.print"):
                flake_gate.main(["detect"])
            return output.read_text(), summary.read_text()

    def test_hands_the_changed_test_files_to_the_jobs(self):
        output, summary = self.detect(
            [
                "autogpt_platform/backend/backend/data/credit_test.py",
                "autogpt_platform/frontend/src/lib/keyboard.test.ts",
                "autogpt_platform/frontend/src/lib/keyboard.ts",
            ]
        )
        self.assertEqual(
            output.splitlines(),
            [
                'backend=["autogpt_platform/backend/backend/data/credit_test.py"]',
                'frontend=["autogpt_platform/frontend/src/lib/keyboard.test.ts"]',
                "self_test=false",
            ],
        )
        self.assertIn("Re-running 1 backend and 1 frontend test files", summary)

    def test_a_change_without_tests_runs_nothing(self):
        output, summary = self.detect(["README.md", ".github/scripts/flake_gate.py"])
        self.assertEqual(
            output.splitlines(), ["backend=[]", "frontend=[]", "self_test=true"]
        )
        self.assertIn("nothing to re-run", summary)


@unittest.skipUnless(Path("/proc/self/stat").exists(), "reads /proc")
class CleanupTests(unittest.TestCase):
    """A run leaves nothing of its own running for the next run to trip on."""

    def leftover_after(self, script: str, timeout: float) -> int:
        with tempfile.TemporaryDirectory() as tmp:
            pid_file = Path(tmp) / "child.pid"
            # A child that ignores SIGTERM, as a service can.
            child = f"(trap '' TERM; exec sleep 300) & echo $! > {pid_file}; "
            suite = BackendSuite(
                prefix=["sh", "-c", child + script], directory=Path(tmp), options=[]
            )
            flake_gate.execute(
                suite,
                ["x_test.py"],
                index=0,
                seed=None,
                pressure=None,
                timeout=timeout,
                out_dir=Path(tmp),
            )
            return int(pid_file.read_text())

    def assert_gone(self, pid: int) -> None:
        for _ in range(100):
            try:
                state = (
                    Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
                )
            except (FileNotFoundError, ProcessLookupError):
                return
            if state == "Z":
                return
            time.sleep(0.05)
        self.fail(f"process {pid} is still running")

    def test_after_a_run_that_exits(self):
        self.assert_gone(self.leftover_after("exit 0", timeout=60))

    def test_after_a_run_that_is_stopped(self):
        self.assert_gone(self.leftover_after("sleep 300", timeout=1))


class LoadTests(unittest.TestCase):
    def test_starts_busy_processes_and_stops_them(self):
        with flake_gate.Load(Pressure(load=2)) as load:
            procs = list(load._procs)
            self.assertEqual(len(procs), 2)
            self.assertTrue(all(p.poll() is None for p in procs))
        self.assertTrue(all(p.poll() is not None for p in procs))

    @unittest.skipUnless(hasattr(os, "sched_getaffinity"), "Linux only")
    def test_holds_the_tests_and_the_load_to_the_pressures_cores(self):
        pressure = Pressure(load=1, cores=1)
        (cpu,) = pressure.cpus() or {None}
        command = [
            sys.executable,
            "-c",
            "import os; print(sorted(os.sched_getaffinity(0)))",
        ]
        tests = subprocess.run(
            command, capture_output=True, text=True, preexec_fn=pressure.pin()
        )
        self.assertEqual(tests.stdout.strip(), f"[{cpu}]")
        with flake_gate.Load(pressure) as load:
            (busy,) = load._procs
            self.assertEqual(os.sched_getaffinity(busy.pid), {cpu})
        self.assertIn("sharing 1 core with the tests", pressure.describe())
        self.assertEqual(pressure.flags(), "--load 1 --cores 1")

    def test_each_suite_has_its_own_pressure(self):
        self.assertEqual(FrontendSuite().pressure(), Pressure(load=2, cores=1))
        backend = BackendSuite().pressure()
        self.assertEqual(backend.cores, 0)
        self.assertEqual(backend.load, 3 * (os.cpu_count() or 1))


if __name__ == "__main__":
    unittest.main()
