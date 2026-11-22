#!/usr/bin/env python3
"""Flake gate: re-run the test files a pull request adds or changes, many
times, under CPU load and in random order, and report the tests whose outcome
changes from run to run.

A test can pass its PR checks and still fail in the merge queue or on dev,
where runners are busier and other tests run first. #15427's ``< 1.5 s``
assertion passed PR CI and then failed three merge groups in a row. This gate
looks for that before merge.

It is advisory: Check PR Status ignores its result (``ADVISORY_CHECK_RUNS`` in
.github/workflows/scripts/check_actions_status.py).

    detect                  list the changed test files (for the workflow)
    run --suite S FILE...   re-run the files and report (CI or local)
    stress -- CMD...        run CMD once under the gate's CPU load (local repro)

Each ``run`` starts with one run in file order without load. A test that fails there too is a real failure, or needs a
service the gate doesn't start; it is listed, not counted. Then come up to
``--runs`` runs under load, each in a new seeded order, while the time budget
lasts. A test that passes in some runs and fails in others is flaky.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parents[1]

DEFAULT_RUNS = 10
DEFAULT_BUDGET_SECONDS = 360
# A loaded run that takes HANG_FACTOR times as long as the run without load,
# and at least HANG_FLOOR_SECONDS, is stuck: it is stopped and counted.
HANG_FACTOR = 10
HANG_FLOOR_SECONDS = 120
# Until a loaded run has finished, assume one takes this many times as long as
# the run without load when deciding whether another fits the budget.
FIRST_LOADED_RUN_FACTOR = 2
MAX_FILES = 40
MAX_ROWS_PER_FILE = 10
MAX_EXCERPTS_PER_FILE = 3
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

GATE_FILES = {
    ".github/workflows/platform-flake-gate.yml",
    ".github/scripts/flake_gate.py",
    ".github/scripts/flake_gate_pytest.py",
    ".github/scripts/test_flake_gate.py",
}

PASSED, FAILED, SKIPPED = "passed", "failed", "skipped"


# ---------------------------------------------------------------- detect


def is_backend_test(path: str) -> bool:
    if not path.startswith("autogpt_platform/backend/") or not path.endswith(".py"):
        return False
    name = path.rsplit("/", 1)[-1]
    return name.startswith("test_") or name.endswith("_test.py")


def is_frontend_test(path: str) -> bool:
    """The files ``vitest.config.mts`` includes."""
    if path.startswith("autogpt_platform/frontend/src/"):
        return path.endswith((".test.ts", ".test.tsx"))
    if path.startswith("autogpt_platform/frontend/scripts/"):
        return path.endswith(".test.ts")
    return False


def changed_files(base: str, head: str) -> list[str]:
    """The files ``head`` adds, modifies or renames relative to ``base``."""
    out = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=AMR", "-z", base, head],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return sorted(name for name in out.split("\0") if name)


@dataclass
class Changes:
    backend: list[str]
    frontend: list[str]
    gate: bool  # the gate itself changed, so its own tests should run


def classify_changes(paths: Iterable[str]) -> Changes:
    paths = list(paths)
    return Changes(
        backend=[p for p in paths if is_backend_test(p)],
        frontend=[p for p in paths if is_frontend_test(p)],
        gate=any(p in GATE_FILES for p in paths),
    )


def cmd_detect(args: argparse.Namespace) -> int:
    changes = classify_changes(changed_files(args.base, args.head))
    print(json.dumps(changes.__dict__, indent=2))
    _append_env_file(
        "GITHUB_OUTPUT",
        f"backend={json.dumps(changes.backend)}\n"
        f"frontend={json.dumps(changes.frontend)}\n"
        f"self_test={'true' if changes.gate else 'false'}\n",
    )
    if changes.backend or changes.frontend:
        lines = [
            "### Flake gate: changed tests",
            "",
            f"Re-running {len(changes.backend)} backend and "
            f"{len(changes.frontend)} frontend test files under CPU load; "
            "the `flake gate (backend)` and `flake gate (frontend)` jobs "
            "report the results.",
            "",
            *[f"- `{p}`" for p in [*changes.backend, *changes.frontend]],
        ]
    else:
        lines = [
            "### Flake gate: no changed tests",
            "",
            "This change adds or modifies no backend or frontend test files, "
            "so there is nothing to re-run.",
        ]
    _append_env_file("GITHUB_STEP_SUMMARY", "\n".join(lines) + "\n")
    return 0


# ---------------------------------------------------------------- suites


@dataclass
class Outcome:
    status: str
    message: str = ""
    line: Optional[int] = None


class Suite:
    name = ""
    directory = Path()
    report_suffix = ""

    def __init__(
        self,
        prefix: Optional[list[str]] = None,
        directory: Optional[Path] = None,
    ):
        self.prefix = prefix or self.default_prefix()
        if directory is not None:
            self.directory = directory

    def default_prefix(self) -> list[str]:
        raise NotImplementedError

    def command(self, files: list[str], seed: Optional[int], report: Path) -> list[str]:
        raise NotImplementedError

    def env(self, seed: Optional[int], report: Path) -> dict[str, str]:
        return dict(os.environ)

    def parse(self, report: Path) -> dict[str, Outcome]:
        """Per-test outcomes keyed by test id. A test id starts with its
        file's path relative to the suite's directory; an id that is just
        the path stands for the file as a whole."""
        raise NotImplementedError

    def file_of(self, test: str, files: list[str]) -> str:
        return max((f for f in files if test.startswith(f)), key=len, default=test)

    def pressure(self) -> Pressure:
        """The load a loaded run of this suite runs under."""
        raise NotImplementedError

    def repro(self, file: str, seed: Optional[int], pressure: Pressure) -> str:
        """The command that repeats a run: under load in the seed's order, or
        as usual when the run that failed was the one without load."""
        raise NotImplementedError


class BackendSuite(Suite):
    name = "backend"
    directory = REPO_ROOT / "autogpt_platform" / "backend"
    report_suffix = "jsonl"

    # Without the test job's coverage: tracing the backend's imports and
    # saving its data takes most of a run's time (43 s of 63 s for a file
    # whose tests take 2 s), so it would leave room for a run or two, and it
    # made no difference to the tests it was tried on.
    OPTIONS = ["-c", "pyproject.toml"]

    def __init__(
        self,
        prefix: Optional[list[str]] = None,
        directory: Optional[Path] = None,
        options: Optional[list[str]] = None,
    ):
        super().__init__(prefix, directory)
        self.options = self.OPTIONS if options is None else options

    def default_prefix(self) -> list[str]:
        return ["poetry", "run", "pytest"]

    def pressure(self) -> Pressure:
        # Three busy processes per core, everywhere: the tests start
        # services in processes of their own, which need cores to run on.
        # It slows a run about 3.5 times, more than coverage slows CI's.
        return Pressure(load=3 * (os.cpu_count() or 1))

    def command(self, files: list[str], seed: Optional[int], report: Path) -> list[str]:
        cmd = [
            *self.prefix,
            *self.options,
            "-p",
            "no:cacheprovider",
            "-p",
            "flake_gate_pytest",
            f"--flake-gate-report={report}",
            "--continue-on-collection-errors",
            "-q",
        ]
        if seed is not None:
            cmd.append(f"--flake-gate-seed={seed}")
        return [*cmd, *files]

    def env(self, seed: Optional[int], report: Path) -> dict[str, str]:
        env = _with_plugin_path(dict(os.environ))
        if seed is not None:
            # Set iteration order follows the hash seed, so pinning it lets a
            # failing run be repeated exactly.
            env["PYTHONHASHSEED"] = str(seed)
        return env

    def parse(self, report: Path) -> dict[str, Outcome]:
        outcomes: dict[str, Outcome] = {}
        started: dict[str, Optional[int]] = {}
        try:
            lines = report.read_text(encoding="utf-8").splitlines()
        except OSError:
            return outcomes
        for raw in lines:
            try:
                record = json.loads(raw)
            except ValueError:
                continue  # a killed run can leave its last line cut off
            nodeid, phase = record["nodeid"], record["phase"]
            if phase == "start":
                started[nodeid] = record.get("line")
                continue
            current = outcomes.get(nodeid)
            outcome = record["outcome"]
            if outcome == FAILED:
                if current is None or current.status != FAILED:
                    outcomes[nodeid] = Outcome(
                        FAILED, record.get("message", ""), started.get(nodeid)
                    )
            elif current is None or (current.status == SKIPPED and outcome == PASSED):
                outcomes[nodeid] = Outcome(outcome, line=started.get(nodeid))
        for nodeid, line in started.items():
            if nodeid not in outcomes:
                outcomes[nodeid] = Outcome(
                    FAILED,
                    "The test never finished: the run was stopped, or the "
                    "test process died, while it ran.",
                    line,
                )
        return outcomes

    def repro(self, file: str, seed: Optional[int], pressure: Pressure) -> str:
        if seed is None:
            return f"poetry run test {shlex.quote(file)}"
        return (
            f"PYTHONHASHSEED={seed} python3 ../../.github/scripts/flake_gate.py "
            f"stress {pressure.flags()} -- poetry run test -p flake_gate_pytest "
            f"--flake-gate-seed={seed} {shlex.quote(file)}"
        )


class FrontendSuite(Suite):
    name = "frontend"
    directory = REPO_ROOT / "autogpt_platform" / "frontend"
    report_suffix = "json"

    def default_prefix(self) -> list[str]:
        return ["pnpm", "exec", "vitest", "run"]

    def pressure(self) -> Pressure:
        # Vitest and its worker held to one core with two busy processes:
        # how W26 reproduced the tests that ejected merge groups (3 of 3 runs
        # failed) while their fixes held (10 of 10). Spreading the same load
        # over a runner's cores didn't fail them: the worker loads modules
        # through the main process, and on one core every one of those calls
        # waits its turn.
        return Pressure(load=2, cores=1)

    def command(self, files: list[str], seed: Optional[int], report: Path) -> list[str]:
        # Coverage on, as the integration test job runs Vitest, so the tests
        # run as slowly as there. Its report covers only these files: one of
        # every source file takes longer to build than most test files take
        # to run, and nothing reads it.
        cmd = [
            *self.prefix,
            *files,
            "--passWithNoTests",
            "--coverage",
            "--coverage.reporter=json-summary",
            f"--coverage.reportsDirectory={report.with_suffix('')}-coverage",
            *(f"--coverage.include={file}" for file in files),
            "--reporter=default",
            "--reporter=json",
            f"--outputFile.json={report}",
        ]
        if seed is not None:
            cmd += ["--sequence.shuffle", f"--sequence.seed={seed}"]
        return cmd

    def parse(self, report: Path) -> dict[str, Outcome]:
        outcomes: dict[str, Outcome] = {}
        try:
            data = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return outcomes
        for result in data.get("testResults", []):
            file = _relative_to(result.get("name", ""), self.directory)
            assertions = result.get("assertionResults", [])
            for assertion in assertions:
                title = " > ".join(
                    [*assertion.get("ancestorTitles", []), assertion["title"]]
                )
                status = assertion.get("status")
                outcomes[f"{file} > {title}"] = Outcome(
                    status if status in (PASSED, FAILED) else SKIPPED,
                    "\n".join(assertion.get("failureMessages") or []),
                )
            if result.get("status") == FAILED and not any(
                a.get("status") == FAILED for a in assertions
            ):
                # The file failed outside any test: it didn't load, or a hook
                # around its tests failed.
                outcomes[file] = Outcome(FAILED, result.get("message", ""))
        return outcomes

    def repro(self, file: str, seed: Optional[int], pressure: Pressure) -> str:
        if seed is None:
            return f"pnpm exec vitest run {shlex.quote(file)} --coverage"
        return (
            f"python3 ../../.github/scripts/flake_gate.py stress {pressure.flags()} "
            f"-- pnpm exec vitest run {shlex.quote(file)} --coverage "
            f"--sequence.shuffle --sequence.seed={seed}"
        )


SUITES: dict[str, type[Suite]] = {"backend": BackendSuite, "frontend": FrontendSuite}


def _with_plugin_path(env: dict[str, str]) -> dict[str, str]:
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(SCRIPTS_DIR), env.get("PYTHONPATH", "")) if p
    )
    return env


def _relative_to(path: str, directory: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(directory.resolve()))
    except ValueError:
        return path


# ---------------------------------------------------------------- load


@dataclass(frozen=True)
class Pressure:
    """What a loaded run runs under: ``load`` busy processes and, when
    ``cores`` is set, the tests and those processes held to that many CPUs."""

    load: int
    cores: int = 0

    def cpus(self) -> Optional[set[int]]:
        if not self.cores or not hasattr(os, "sched_getaffinity"):
            return None
        return set(sorted(os.sched_getaffinity(0))[: self.cores])

    def pin(self) -> Optional[Callable[[], None]]:
        """A ``preexec_fn`` that holds a process to ``cpus()``."""
        cpus = self.cpus()
        if cpus is None:
            return None
        return lambda: os.sched_setaffinity(0, cpus)

    def describe(self) -> str:
        if self.cpus() is not None:
            return f"{_processes(self.load)} sharing {_count(self.cores, 'core')} with the tests"
        return f"{_processes(self.load)} on {_count(os.cpu_count() or 1, 'core')}"

    def flags(self) -> str:
        return f"--load {self.load}" + (f" --cores {self.cores}" if self.cores else "")


class Load:
    """The pressure's busy processes, for as long as the context is open."""

    def __init__(self, pressure: Optional[Pressure]):
        self.pressure = pressure
        self._procs: list[subprocess.Popen] = []

    def __enter__(self) -> "Load":
        if self.pressure is None:
            return self
        self._procs = [
            subprocess.Popen(
                [sys.executable, "-c", "while True: pass"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=self.pressure.pin(),
            )
            for _ in range(self.pressure.load)
        ]
        return self

    def __exit__(self, *exc: object) -> None:
        for proc in self._procs:
            proc.kill()
        for proc in self._procs:
            proc.wait()


# ---------------------------------------------------------------- runs


@dataclass
class Run:
    index: int
    seed: Optional[int]  # None for the run in file order without load
    load: int
    duration: float
    exit_code: Optional[int]  # None when the run was stopped
    stopped_after: Optional[float]
    reported: bool  # the test runner wrote a report
    outcomes: dict[str, Outcome]
    log: Path

    @property
    def label(self) -> str:
        return "without load" if self.seed is None else f"seed {self.seed}"


def execute(
    suite: Suite,
    files: list[str],
    *,
    index: int,
    seed: Optional[int],
    pressure: Optional[Pressure],
    timeout: float,
    out_dir: Path,
) -> Run:
    report = out_dir / f"run-{index}.{suite.report_suffix}"
    log = out_dir / f"run-{index}.log"
    report.unlink(missing_ok=True)
    cmd = suite.command(files, seed, report)
    stopped_after = None
    with Load(pressure), log.open("w", encoding="utf-8") as out:
        out.write(f"$ {shlex.join(cmd)}\n\n")
        out.flush()
        started = time.monotonic()
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=suite.directory,
                env=suite.env(seed, report),
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                preexec_fn=pressure.pin() if pressure else None,
            )
        except OSError as e:
            out.write(f"Could not start the tests: {e}\n")
            exit_code: Optional[int] = 127
        else:
            try:
                exit_code = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                stopped_after = time.monotonic() - started
                _stop(proc)
                exit_code = None
            else:
                _kill_group(proc)
        duration = time.monotonic() - started
    outcomes = suite.parse(report)
    reported = bool(outcomes) or report.exists()
    _add_file_outcomes(suite, files, outcomes, exit_code, stopped_after)
    load = pressure.load if pressure else 0
    return Run(
        index, seed, load, duration, exit_code, stopped_after, reported, outcomes, log
    )


def _stop(proc: subprocess.Popen) -> None:
    """Stop a run that is out of time: ask it to end, then end what's left."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    _kill_group(proc)


def _kill_group(proc: subprocess.Popen) -> None:
    """Kill whatever is left of a run's process group: a service its tests
    started can outlive them, and must not hold a port, a connection or a CPU
    while the next run runs."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def _add_file_outcomes(
    suite: Suite,
    files: list[str],
    outcomes: dict[str, Outcome],
    exit_code: Optional[int],
    stopped_after: Optional[float],
) -> None:
    """Give each file an outcome of its own, so a run that crashed or hung
    before a file reported anything counts against that file."""
    seen = {suite.file_of(test, files) for test in outcomes}
    for file in files:
        if file in outcomes:
            continue  # the file itself failed: it didn't collect or load
        if file in seen:
            outcomes[file] = Outcome(PASSED)
        elif stopped_after is not None:
            outcomes[file] = Outcome(
                FAILED,
                f"The run was stopped after {stopped_after:.0f} s, before this "
                "file reported any result.",
            )
        elif exit_code not in (0, 1, 5):
            # 5: pytest collected no tests. Any other code with no results
            # means the run never reached the tests.
            outcomes[file] = Outcome(
                FAILED,
                f"The run exited with code {exit_code} before this file "
                "reported any result.",
            )


@dataclass
class GateResult:
    suite: Suite
    files: list[str]
    left_out: list[str]
    budget: float
    pressure: Pressure
    runs: list[Run] = field(default_factory=list)
    elapsed: float = 0.0
    note: str = ""
    error: bool = False  # the gate could not run the tests at all
    verdicts: dict[str, "Verdict"] = field(default_factory=dict)

    def of_kind(self, kind: str) -> list["Verdict"]:
        return [v for v in self.verdicts.values() if v.kind == kind]


def run_gate(
    suite: Suite,
    files: list[str],
    *,
    runs: int,
    budget: float,
    pressure: Pressure,
    seed: Optional[int],
    out_dir: Path,
    clock: Callable[[], float] = time.monotonic,
    log: Callable[[str], None] = lambda line: print(line, flush=True),
) -> GateResult:
    files = sorted({_suite_relative(f, suite) for f in files})
    result = GateResult(suite, files[:MAX_FILES], files[MAX_FILES:], budget, pressure)
    files = result.files
    out_dir.mkdir(parents=True, exist_ok=True)
    started = clock()
    deadline = started + budget

    baseline = execute(
        suite, files, index=0, seed=None, pressure=None, timeout=budget, out_dir=out_dir
    )
    log(_progress(baseline))
    result.runs.append(baseline)
    tests_run = [t for t in baseline.outcomes if t not in files]
    if baseline.stopped_after is not None:
        result.note = (
            f"The changed tests took more than the {budget:.0f} s budget even "
            "without load, so they were not re-run."
        )
    elif not baseline.reported:
        if baseline.exit_code in (0, 5):
            result.note = "The changed files have no tests to run."
        else:
            result.error = True
            result.note = (
                f"The gate could not run the tests (exit code "
                f"{baseline.exit_code}); see `{baseline.log.name}` in the "
                "job's artifact."
            )
    elif not tests_run:
        result.note = (
            "No test in the changed files ran: they failed to load, so they "
            "were not re-run."
        )
    else:
        hang_after = max(HANG_FLOOR_SECONDS, HANG_FACTOR * baseline.duration)
        estimate = FIRST_LOADED_RUN_FACTOR * baseline.duration
        if seed is None:
            seed = random.SystemRandom().randrange(1, 2**31 - runs)
        for i in range(runs):
            remaining = deadline - clock()
            if estimate > remaining:
                result.note = f"{i} of {runs} loaded runs fit the budget."
                break
            timeout = min(hang_after, remaining)
            run = execute(
                suite,
                files,
                index=i + 1,
                seed=seed + i,
                pressure=pressure,
                timeout=timeout,
                out_dir=out_dir,
            )
            log(_progress(run))
            if run.stopped_after is not None and timeout < hang_after:
                result.note = (
                    f"Loaded run {i + 1} ran out of budget and was not counted; "
                    f"{i} of {runs} loaded runs finished."
                )
                break
            result.runs.append(run)
            estimate = max(r.duration for r in result.runs[1:])

    result.elapsed = clock() - started
    result.verdicts = classify(suite, files, result.runs)
    return result


def _suite_relative(path: str, suite: Suite) -> str:
    prefix = _repo_path(suite.directory) + "/"
    return path[len(prefix) :] if path.startswith(prefix) else path


def _progress(run: Run) -> str:
    counts = {status: 0 for status in (PASSED, FAILED, SKIPPED)}
    for outcome in run.outcomes.values():
        counts[outcome.status] += 1
    stopped = (
        f", stopped after {run.stopped_after:.0f} s"
        if run.stopped_after is not None
        else ""
    )
    return (
        f"run {run.index} ({run.label}, load {run.load}): "
        f"{counts[PASSED]} passed, {counts[FAILED]} failed, "
        f"{counts[SKIPPED]} skipped in {run.duration:.1f} s{stopped}"
    )


# ---------------------------------------------------------------- verdicts


@dataclass
class Verdict:
    test: str
    file: str
    kind: str  # "flaky", "broken", "stable" or "skipped"
    passes: list[Run]
    failures: list[tuple[Run, Outcome]]

    @property
    def line(self) -> Optional[int]:
        return next((o.line for _, o in self.failures if o.line), None)


def classify(suite: Suite, files: list[str], runs: list[Run]) -> dict[str, Verdict]:
    verdicts = {}
    for test in sorted({test for run in runs for test in run.outcomes}):
        passes = [r for r in runs if _status(r, test) == PASSED]
        failures = [(r, r.outcomes[test]) for r in runs if _status(r, test) == FAILED]
        if passes and failures:
            kind = "flaky"
        elif failures:
            kind = "broken"
        elif passes:
            kind = "stable"
        else:
            kind = "skipped"
        verdicts[test] = Verdict(
            test, suite.file_of(test, files), kind, passes, failures
        )
    return verdicts


def _status(run: Run, test: str) -> Optional[str]:
    outcome = run.outcomes.get(test)
    return outcome.status if outcome else None


# ---------------------------------------------------------------- report


def render(result: GateResult) -> str:
    flaky = result.of_kind("flaky")
    loaded = [r for r in result.runs if r.seed is not None]
    if result.error:
        headline = "⚠️ could not run the tests"
    elif flaky:
        headline = f"❌ {_count(len(flaky), 'flaky test')}"
    elif loaded:
        headline = "✅ no flaky tests"
    else:
        headline = "not re-run"
    lines = [
        f"### Flake gate ({result.suite.name}): {headline}",
        "",
        f"Re-ran {_count(len(result.files), 'changed test file')} once in file "
        f"order without load, then {_count(len(loaded), 'time')} under CPU load "
        f"({result.pressure.describe()}), each "
        f"in a new random order: {_duration(result.elapsed)} of a "
        f"{_duration(result.budget)} budget.",
    ]
    if result.note:
        lines += ["", result.note]
    if result.left_out:
        lines += [
            "",
            f"Only the first {MAX_FILES} files were re-run; "
            f"{len(result.left_out)} more were left out.",
        ]
    lines += _flaky_section(result)
    lines += _broken_section(result)
    stable = [v for v in result.of_kind("stable") if v.test not in result.files]
    if stable:
        lines += ["", f"**Stable**: {_count(len(stable), 'test')} passed in every run."]
    return "\n".join(lines) + "\n"


def _flaky_section(result: GateResult) -> list[str]:
    by_file = _by_file(result.of_kind("flaky"))
    if not by_file:
        return []
    lines = [
        "",
        "**Flaky**: passed in some runs and failed in others. CI runners get "
        "this busy, and other tests change the order, so expect these to fail "
        "in the merge queue or on dev. Assert what the code does (call counts, "
        "order, state) rather than how fast it does it, and wait for events "
        "rather than for time to pass.",
    ]
    for file, verdicts in by_file.items():
        lines += ["", f"`{file}`", "", "| Test | Failed | Failing runs |"]
        lines.append("| --- | --- | --- |")
        for verdict in verdicts[:MAX_ROWS_PER_FILE]:
            total = len(verdict.passes) + len(verdict.failures)
            unloaded = any(r.seed is None for r in verdict.passes)
            failing = ", ".join(r.label for r, _ in verdict.failures[:5])
            if len(verdict.failures) > 5:
                failing += ", …"
            lines.append(
                f"| {_cell(_test_name(verdict))} | {len(verdict.failures)} of "
                f"{total} runs{' (passed without load)' if unloaded else ''} "
                f"| {failing} |"
            )
        if len(verdicts) > MAX_ROWS_PER_FILE:
            lines.append(f"| … and {len(verdicts) - MAX_ROWS_PER_FILE} more | | |")

    lines += ["", "<details><summary>First failures</summary>", ""]
    for verdicts in by_file.values():
        for verdict in verdicts[:MAX_EXCERPTS_PER_FILE]:
            run, outcome = verdict.failures[0]
            lines += [
                f"`{verdict.test}` ({run.label}):",
                "",
                "```",
                _excerpt(outcome.message) or "(no message)",
                "```",
                "",
            ]
    lines += ["</details>", ""]

    suite = result.suite
    where = f"`{_repo_path(suite.directory)}`"
    if suite.name == "backend":
        where += " (`poetry run test` brings the test stack up)"
    lines += [f"**Reproduce** a failing run, from {where}:", "", "```bash"]
    for file, verdicts in by_file.items():
        failed = [r for v in verdicts for r, _ in v.failures]
        loaded = next((r for r in failed if r.seed is not None), failed[0])
        lines.append(suite.repro(file, loaded.seed, result.pressure))
    lines += [
        "```",
        "",
        "To repeat the whole gate on a file, see the flake gate section of "
        f"`{_repo_path(suite.directory)}/TESTING.md`.",
    ]
    return lines


def _broken_section(result: GateResult) -> list[str]:
    broken = result.of_kind("broken")
    if not broken:
        return []
    lines = [
        "",
        "**Not counted**: failed in every run, including the one without load. "
        "That is a real failure, which the regular CI reports too, or a test "
        "that needs something this gate doesn't provide (ClamAV, FalkorDB, API "
        "keys).",
        "",
    ]
    lines += [f"- `{v.test}`" for v in broken[:20]]
    if len(broken) > 20:
        lines.append(f"- … and {len(broken) - 20} more")
    return lines


def annotation_commands(result: GateResult) -> list[str]:
    """One error annotation per file with flaky tests, on the PR's diff."""
    out = []
    for file, verdicts in _by_file(result.of_kind("flaky")).items():
        path = _repo_path(result.suite.directory / file)
        props = f"file={_escape_property(path)}"
        line = next((v.line for v in verdicts if v.line), None)
        if line:
            props += f",line={line}"
        props += ",title=Flaky under CPU load"
        names = "; ".join(_test_name(v) for v in verdicts[:3])
        if len(verdicts) > 3:
            names += f"; and {len(verdicts) - 3} more"
        message = (
            f"{_count(len(verdicts), 'test')} passed in some runs and failed in "
            f"others, under CPU load and in random order: {names}. The flake "
            "gate job summary says how to reproduce it."
        )
        out.append(f"::error {props}::{_escape_data(message)}")
    return out


def _by_file(verdicts: list[Verdict]) -> dict[str, list[Verdict]]:
    by_file: dict[str, list[Verdict]] = {}
    for verdict in verdicts:
        by_file.setdefault(verdict.file, []).append(verdict)
    return by_file


def _test_name(verdict: Verdict) -> str:
    """The test's id without its file, which a heading or path already says."""
    if verdict.test == verdict.file:
        return "(the file as a whole)"
    rest = verdict.test[len(verdict.file) :]
    for separator in ("::", " > "):
        if rest.startswith(separator):
            return rest[len(separator) :]
    return verdict.test


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


def _processes(n: int) -> str:
    return f"{n} busy process{'' if n == 1 else 'es'}"


def _duration(seconds: float) -> str:
    minutes, secs = divmod(int(round(seconds)), 60)
    return f"{minutes} m {secs:02d} s" if minutes else f"{secs} s"


def _excerpt(message: str, limit: int = 12) -> str:
    # Testing Library follows its message with a dump of the whole DOM.
    text = _ANSI_ESCAPE.sub("", message).split("\nIgnored nodes:", 1)[0]
    lines = text.strip().splitlines()
    # pytest marks the lines that explain a failure with "E". Vitest puts the
    # message first and a stack of "at ..." frames after it.
    errors = [line for line in lines if line.startswith("E ")]
    if not errors:
        errors = [line for line in lines if not line.lstrip().startswith("at ")]
    picked = (errors or lines)[:limit]
    return "\n".join(line[:300] for line in picked).replace("```", "'''")


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("`", "'")


def _repo_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _escape_data(text: str) -> str:
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _escape_property(text: str) -> str:
    return _escape_data(text).replace(":", "%3A").replace(",", "%2C")


def _append_env_file(variable: str, text: str) -> None:
    path = os.environ.get(variable)
    if path:
        with open(path, "a", encoding="utf-8") as out:
            out.write(text)


def cmd_run(args: argparse.Namespace) -> int:
    files = list(args.files)
    if args.files_from_env:
        files += json.loads(os.environ.get(args.files_from_env) or "[]")
    if not files:
        print("No test files to re-run.")
        return 0
    suite_type = SUITES[args.suite]
    suite = suite_type(shlex.split(args.command)) if args.command else suite_type()
    pressure = suite.pressure()
    pressure = Pressure(
        load=pressure.load if args.load is None else args.load,
        cores=pressure.cores if args.cores is None else args.cores,
    )
    out_dir = Path(args.out_dir).resolve()
    result = run_gate(
        suite,
        files,
        runs=args.runs,
        budget=args.budget,
        pressure=pressure,
        seed=args.seed,
        out_dir=out_dir,
    )
    report = render(result)
    (out_dir / "summary.md").write_text(report, encoding="utf-8")
    _append_env_file("GITHUB_STEP_SUMMARY", report)
    print()
    print(report)
    if os.environ.get("GITHUB_ACTIONS") == "true":
        for line in annotation_commands(result):
            print(line)
    return 1 if result.error or result.of_kind("flaky") else 0


def cmd_stress(args: argparse.Namespace) -> int:
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        print(
            "usage: flake_gate.py stress [--load N] [--cores N] -- COMMAND...",
            file=sys.stderr,
        )
        return 2
    pressure = Pressure(load=args.load, cores=args.cores)
    print(f"Under CPU load ({pressure.describe()}): {shlex.join(command)}")
    with Load(pressure):
        env = _with_plugin_path(dict(os.environ))
        return subprocess.run(command, env=env, preexec_fn=pressure.pin()).returncode


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    detect = sub.add_parser("detect", help="list the changed test files")
    detect.add_argument("--base", default="HEAD^1")
    detect.add_argument("--head", default="HEAD")
    detect.set_defaults(func=cmd_detect)

    run = sub.add_parser("run", help="re-run test files and report flaky tests")
    run.add_argument("--suite", choices=sorted(SUITES), required=True)
    run.add_argument("files", nargs="*", help="test files, repo- or suite-relative")
    run.add_argument(
        "--files-from-env",
        metavar="VAR",
        help="also re-run the files in this environment variable's JSON list",
    )
    run.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="loaded runs")
    run.add_argument(
        "--budget",
        type=float,
        default=float(
            os.environ.get("FLAKE_GATE_BUDGET_SECONDS", DEFAULT_BUDGET_SECONDS)
        ),
        help="seconds for all the runs together",
    )
    run.add_argument(
        "--load",
        type=int,
        help="busy processes during a loaded run (default: the suite's)",
    )
    run.add_argument(
        "--cores",
        type=int,
        help="hold the tests and the busy processes to this many cores, 0 for "
        "none (default: the suite's)",
    )
    run.add_argument("--seed", type=int, help="seed of the first loaded run")
    run.add_argument(
        "--command",
        help="run the tests with this command instead of the suite's own, "
        "e.g. 'poetry run test' to bring the backend test stack up",
    )
    run.add_argument("--out-dir", default="flake-gate-results")
    run.set_defaults(func=cmd_run)

    stress = sub.add_parser("stress", help="run a command once under CPU load")
    stress.add_argument("--load", type=int, default=3 * (os.cpu_count() or 1))
    stress.add_argument("--cores", type=int, default=0)
    stress.add_argument("command", nargs=argparse.REMAINDER)
    stress.set_defaults(func=cmd_stress)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
