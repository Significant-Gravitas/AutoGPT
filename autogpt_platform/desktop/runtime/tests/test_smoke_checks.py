"""The smoke test's own checks on the bundle and on what it leaves behind.

They only bite on a real stack; these make sure they would: that a new
directory, a leftover of an earlier run and a refused write are each seen,
and that the temporary data directory really goes.
"""

import importlib
import io
import json
import os
import signal
import stat
import subprocess
import sys
import threading
import time
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import psutil
import pytest

DESKTOP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DESKTOP / "build"))
smoke_test = importlib.import_module("smoke_test")


SAFE_PREFIX = "DESKTOP_SMOKE_DIAGNOSTIC "
RUNTIME_PREFIX = "DESKTOP_SMOKE_RUNTIME_DIAGNOSTIC "
PRIVATE_CANARY = "secret-token C:/private/settings.env bearer=do-not-upload"


@pytest.mark.parametrize(
    "message,error,process,code,os_error",
    [
        ("startup was cancelled", "startup_cancelled", "unknown", None, "none"),
        (
            "the other side of this start failed",
            "startup_abandoned",
            "unknown",
            None,
            "none",
        ),
        (
            "the database did not start",
            "database_not_started",
            "postgres",
            None,
            "none",
        ),
        (
            f"workers exited while starting. See {PRIVATE_CANARY} for details.",
            "process_exited_starting",
            "workers",
            None,
            "none",
        ),
        (
            f"valkey did not start. See {PRIVATE_CANARY} for details.",
            "process_not_started",
            "valkey",
            None,
            "none",
        ),
        (
            f"api cannot run with this version of the backend. See {PRIVATE_CANARY}.",
            "backend_contract",
            "api",
            None,
            "none",
        ),
        (
            f"workers keeps stopping (exit code 78). See {PRIVATE_CANARY}.",
            "process_restart_limit",
            "workers",
            78,
            "none",
        ),
        (
            f"unknown-private-process keeps stopping (exit code 3). See {PRIVATE_CANARY}.",
            "unknown",
            "unknown",
            None,
            "none",
        ),
        (
            f"workers keeps stopping (exit code 4294967296). See {PRIVATE_CANARY}.",
            "unknown",
            "unknown",
            None,
            "none",
        ),
        (
            "AutoGPT did not become ready in time: no answer from rest, frontend",
            "apps_not_ready",
            "unknown",
            None,
            "none",
        ),
        (
            f"AutoGPT did not become ready in time: no answer from {PRIVATE_CANARY}",
            "unknown",
            "unknown",
            None,
            "none",
        ),
        (
            f"initdb failed:\n{PRIVATE_CANARY}",
            "postgres_initdb",
            "postgres",
            None,
            "none",
        ),
        (
            f"prisma migrate deploy failed:\n{PRIVATE_CANARY}",
            "migration_deploy",
            "unknown",
            None,
            "none",
        ),
        (
            "This AutoGPT bundle has no database migration history.",
            "migration_history_missing",
            "unknown",
            None,
            "none",
        ),
        (
            f"a PostgreSQL option may not contain a double quote: {PRIVATE_CANARY}",
            "postgres_option",
            "postgres",
            None,
            "none",
        ),
        (
            f"[WinError 5] {PRIVATE_CANARY}",
            "os_error",
            "unknown",
            None,
            "access_denied",
        ),
        (
            f"[WinError 193] {PRIVATE_CANARY}",
            "os_error",
            "unknown",
            None,
            "invalid_executable",
        ),
        (f"[Errno 2] {PRIVATE_CANARY}", "os_error", "unknown", None, "not_found"),
        (f"[WinError 987654] {PRIVATE_CANARY}", "os_error", "unknown", None, "unknown"),
        (PRIVATE_CANARY, "unknown", "unknown", None, "none"),
    ],
)
def test_runtime_context_fatal_forms_never_echo(
    message, error, process, code, os_error
):
    context = smoke_test.RuntimeContext()
    context.observe(
        {"event": "progress", "step": "services", "message": PRIVATE_CANARY}
    )
    context.observe({"event": "timing", "phase": "database", "seconds": 1.5})
    context.observe({"event": "error", "fatal": True, "message": message})
    record = context.record(1, "ready", smoke_test.StopObservation())
    assert record["progress"] == "services"
    assert record["phase"] == "database"
    assert record["fatal"] is True
    assert (
        record["error"],
        record["process"],
        record["process_exit"],
        record["os_error"],
    ) == (error, process, code, os_error)
    assert PRIVATE_CANARY not in json.dumps(record)


@pytest.mark.parametrize(
    "event",
    [
        {"event": "error", "fatal": 1, "message": PRIVATE_CANARY},
        {
            "event": "error",
            "fatal": True,
            "message": PRIVATE_CANARY,
            "extra": PRIVATE_CANARY,
        },
        {"event": "progress", "step": [], "message": PRIVATE_CANARY},
        {"event": "timing", "phase": "database", "seconds": True},
        {"event": "timing", "phase": "database", "seconds": -1},
        {"event": "timing", "phase": "database", "seconds": float("nan")},
        {"event": "timing", "phase": "database", "seconds": 3601},
        {
            "event": "timing",
            "phase": "total",
            "seconds": 2,
            "phases": {PRIVATE_CANARY: 1},
        },
    ],
)
def test_runtime_context_rejects_untrusted_event_shapes(event):
    context = smoke_test.RuntimeContext()
    context.observe(event)
    record = context.record(1, "ready", smoke_test.StopObservation())
    assert record["events_rejected"] is True
    assert record["fatal"] is False
    assert PRIVATE_CANARY not in json.dumps(record)


def test_runtime_context_unknown_phase_and_progress_are_closed():
    context = smoke_test.RuntimeContext()
    context.observe(
        {"event": "progress", "step": PRIVATE_CANARY, "message": PRIVATE_CANARY}
    )
    context.observe({"event": "timing", "phase": PRIVATE_CANARY, "seconds": 1})
    record = context.record(1, "ready", smoke_test.StopObservation())
    assert record["progress"] == record["phase"] == "unknown"
    assert PRIVATE_CANARY not in json.dumps(record)


@pytest.mark.parametrize(
    "payload",
    [
        '{"event":"error","fatal":false,"fatal":true,"message":"secret"}\n',
        '{"event":"error","fatal":true,"message":"' + "x" * 16384 + '"}\n',
        "[\n",
        "[]\n",
    ],
)
def test_runtime_context_reader_rejects_duplicate_overlong_and_malformed(
    payload, tmp_path
):
    process = SimpleNamespace(stdout=io.StringIO(payload))
    said = smoke_test.Said(process, time.monotonic(), smoke_test.Install(False))
    assert said.finished.wait(2)
    record = said.context.record(1, "ready", smoke_test.StopObservation())
    assert record["events_rejected"]
    assert not record["fatal"]


@pytest.mark.parametrize(
    "code,expected", [(0, "clean"), (7, "nonzero"), (-9, "nonzero"), (None, "timeout")]
)
def test_runtime_context_stop_uses_wait_observation_not_poll(code, expected):
    process = Mock()
    process.poll.return_value = None
    process.wait.side_effect = (
        subprocess.TimeoutExpired(PRIVATE_CANARY, 30) if code is None else None
    )
    process.wait.return_value = code
    observation = smoke_test.StopObservation()
    failures = smoke_test.stop(process, observation)
    assert observation.outcome == expected
    assert observation.child_exit == code
    assert bool(failures) == (code != 0)
    process.wait.assert_called_once_with(30)
    assert process.kill.call_count == int(code is None)


class RuntimeDiagnosticSink(io.StringIO):
    def __init__(self, operation):
        super().__init__()
        self.operation, self.target, self.failures = operation, False, 0

    def write(self, text):
        self.target = text.startswith(RUNTIME_PREFIX) or (self.target and text == "\n")
        if self.target and self.operation == "write":
            self.failures += 1
            raise OSError(28, PRIVATE_CANARY)
        return super().write(text)

    def flush(self):
        if self.target and self.operation == "flush":
            self.failures += 1
            raise OSError(28, PRIVATE_CANARY)


@pytest.mark.parametrize("operation", ["write", "flush"])
@pytest.mark.parametrize("primary", [False, True])
def test_runtime_context_output_loss_preserves_cleanup_and_primary(
    diagnostic_runtime, tmp_path, operation, primary
):
    _process, said, leftovers = diagnostic_runtime
    original = TimeoutError(PRIVATE_CANARY)
    if primary:
        said.wait_for_ready.side_effect = original
    sink = RuntimeDiagnosticSink(operation)
    install = smoke_test.Install(False)
    install.start_number = 1
    with redirect_stdout(sink):
        if primary:
            with pytest.raises(TimeoutError) as raised:
                smoke_test.run(tmp_path, tmp_path, 1, lambda *args: [], install)
            assert raised.value is original
        else:
            assert smoke_test.run(tmp_path, tmp_path, 1, lambda *args: [], install)
    assert sink.failures
    leftovers.assert_called_once()


def test_runtime_context_four_start_parity(
    diagnostic_smoke, diagnostic_runtime, monkeypatch, capsys
):
    monkeypatch.setattr(smoke_test, "checked", lambda *args: [])
    assert smoke_test.smoke(*diagnostic_smoke) == 0
    output = capsys.readouterr().out
    records = [
        json.loads(line[len(RUNTIME_PREFIX) :])
        for line in output.splitlines()
        if line.startswith(RUNTIME_PREFIX)
    ]
    assert [(r["start"], r["point"]) for r in records] == [
        (n, point) for n in range(1, 5) for point in ("ready", "stop")
    ]
    assert diagnostic_runtime[2].call_count == 4


def test_runtime_context_existing_v1_allowlists_are_unchanged():
    import installed_smoke_windows as driver

    for field in ("stage", "status", "reason", "exception", "contract"):
        names = {"status": "STATUSES"}
        assert driver.DIAGNOSTIC_ENUMS[field] == getattr(
            smoke_test, "DIAGNOSTIC_" + names.get(field, field.upper() + "S")
        )
    assert smoke_test.DIAGNOSTIC_PREFIX == driver.DIAGNOSTIC_PREFIX == SAFE_PREFIX


def test_runtime_context_bounded_events_and_first_fatal_are_sticky():
    context = smoke_test.RuntimeContext()
    context.observe(
        {"event": "error", "fatal": True, "message": "the database did not start"}
    )
    context.observe({"event": "error", "fatal": False, "message": PRIVATE_CANARY})
    context.observe({"event": "error", "fatal": True, "message": PRIVATE_CANARY})
    for _ in range(smoke_test.MAX_RUNTIME_CONTEXT_EVENTS):
        context.observe(
            {"event": "progress", "step": "services", "message": PRIVATE_CANARY}
        )
    record = context.record(1, "stop", smoke_test.StopObservation("timeout"))
    assert record["error"] == "database_not_started"
    assert record["events_truncated"] is True
    assert record["stop"] == "timeout" and record["stop_exit"] is None


def test_runtime_context_said_first_start_failure_with_observed_stop(
    diagnostic_smoke, diagnostic_runtime, monkeypatch, capsys
):
    process, _, leftovers = diagnostic_runtime
    context = smoke_test.RuntimeContext()
    context.observe(
        {"event": "progress", "step": "infrastructure", "message": PRIVATE_CANARY}
    )
    context.observe(
        {
            "event": "error",
            "fatal": True,
            "message": "initdb failed:\n" + PRIVATE_CANARY,
        }
    )
    monkeypatch.setattr(
        smoke_test,
        "Said",
        lambda *args: SimpleNamespace(
            context=context,
            wait_for_ready=lambda _: None,
            finished=SimpleNamespace(wait=lambda _: True),
        ),
    )
    process.poll.return_value = None
    process.wait.side_effect = subprocess.TimeoutExpired(PRIVATE_CANARY, 30)
    monkeypatch.setattr(smoke_test, "stop", ORIGINAL_STOP)
    assert smoke_test.smoke(*diagnostic_smoke) == 1
    output = capsys.readouterr().out
    records = [
        json.loads(line[len(RUNTIME_PREFIX) :])
        for line in output.splitlines()
        if line.startswith(RUNTIME_PREFIX)
    ]
    assert records[0]["error"] == "postgres_initdb"
    assert records[0]["progress"] == "infrastructure"
    assert records[0]["stop"] == "not_observed"
    assert records[1]["stop"] == "timeout"
    assert records[1]["stop_exit"] is None
    assert PRIVATE_CANARY not in json.dumps(records)
    leftovers.assert_called_once()


ORIGINAL_STOP = smoke_test.stop
ORIGINAL_SAID = smoke_test.Said


class InlineRuntimeReader:
    def __init__(self, target, **kwargs):
        self.target = target

    def start(self):
        self.target()


@pytest.mark.parametrize(
    "tail", ["duplicate", "oversized", "malformed", "nonobject", "unicode", "io_error"]
)
@pytest.mark.parametrize("mode", ["cancelled", "deadline", "hurried"])
def test_runtime_context_lost_cancellation_input_fails_real_stop_checker(
    tail, mode, lifecycle_stop, tmp_path, monkeypatch
):
    install, clock, Process = lifecycle_stop
    install.stopping.cancels = mode == "cancelled"
    install.stopping.hurried = mode == "hurried"
    monkeypatch.setattr(smoke_test.threading, "Thread", Mock())
    process = Process(20, 57)
    cancellation = {
        "event": "shutdown",
        "phase": "cancellation",
        "seconds": {"cancelled": 1, "deadline": 20, "hurried": 3}[mode],
        "outcome": mode,
    }
    extra = {
        "duplicate": json.dumps(cancellation).replace(
            '"event": "shutdown"', '"event": "shutdown", "event": "shutdown"'
        ),
        "oversized": json.dumps({**cancellation, "detail": "x" * 16384}),
        "malformed": "{",
        "nonobject": "[]",
    }.get(tail, "")
    text = (
        json.dumps(
            {
                "event": "progress",
                "step": "stopping",
                "message": "Stopping",
                "grace_seconds": 120,
            }
        )
        + "\n"
        + json.dumps(cancellation)
        + "\n"
        + extra
    )
    stream = io.StringIO(text)
    if tail in {"unicode", "io_error"}:

        def readline(limit):
            line = stream.readline(limit)
            if line:
                return line
            if tail == "unicode":
                raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, PRIVATE_CANARY)
            raise OSError(PRIVATE_CANARY)

        process.stdout = SimpleNamespace(readline=readline)
    else:
        process.stdout = stream
    install.said = ORIGINAL_SAID(process, 0, install)

    def wait(timeout):
        clock.seconds = 57
        install.said.read()
        return 0

    process.wait = wait
    failures = smoke_test.stop_the_runs(process, DESKTOP, tmp_path, install)
    assert failures == [smoke_test.RUNTIME_EVENT_LOSS]
    assert install.said.finished.is_set()
    assert install.said.evidence_lost
    assert not any("one cancellation phase" in failure for failure in failures)
    assert PRIVATE_CANARY not in json.dumps(failures)


@pytest.mark.parametrize(
    "terminal", ["ready", "fatal_event", "stdout_eof", "reader_error", "deadline"]
)
def test_runtime_context_first_terminal_snapshot_is_separate_from_stop(
    terminal, monkeypatch
):
    monkeypatch.setattr(smoke_test.threading, "Thread", Mock())
    process = SimpleNamespace(stdout=io.StringIO(""))
    said = ORIGINAL_SAID(process, time.monotonic(), smoke_test.Install(False))
    said.hear(
        {"event": "progress", "step": "infrastructure", "message": PRIVATE_CANARY}
    )
    said.hear({"event": "timing", "phase": "database", "seconds": 1})
    if terminal == "ready":
        said.hear({"event": "ready", "url": "http://unused.invalid"})
    elif terminal == "fatal_event":
        said.hear({"event": "error", "fatal": True, "message": PRIVATE_CANARY})
    elif terminal == "deadline":
        assert said.wait_for_ready(0) is None
    else:
        if terminal == "reader_error":
            process.stdout = Mock()
            process.stdout.readline.side_effect = UnicodeDecodeError(
                "utf-8", b"\xff", 0, 1, PRIVATE_CANARY
            )
        said.read()
    said.hear(
        {
            "event": "progress",
            "step": "stopping",
            "message": PRIVATE_CANARY,
            "grace_seconds": 120,
        }
    )
    said.hear({"event": "timing", "phase": "queue", "seconds": 2})
    ready = said.diagnostic_context("ready").record(
        1, "ready", smoke_test.StopObservation()
    )
    stopped = said.diagnostic_context("stop").record(
        1, "stop", smoke_test.StopObservation()
    )
    assert ready["settle"] == stopped["settle"] == terminal
    assert (ready["progress"], ready["phase"], ready["completed_phases"]) == (
        "infrastructure",
        "database",
        2,
    )
    assert (stopped["progress"], stopped["phase"], stopped["completed_phases"]) == (
        "stopping",
        "queue",
        10,
    )
    assert PRIVATE_CANARY not in json.dumps([ready, stopped])
    said.diagnostic_context("ready").values["progress"] = "changed copy"
    assert said.diagnostic_context("ready").values["progress"] == "infrastructure"


def test_runtime_context_terminal_lock_order_resolves_timeout_race(monkeypatch):
    monkeypatch.setattr(smoke_test.threading, "Thread", Mock())
    said = ORIGINAL_SAID(
        SimpleNamespace(stdout=io.StringIO("")),
        time.monotonic(),
        smoke_test.Install(False),
    )

    def expired_wait(timeout):
        said.hear({"event": "ready", "url": "http://unused.invalid"})
        return False

    monkeypatch.setattr(said.settled, "wait", expired_wait)
    assert said.wait_for_ready(0) == "http://unused.invalid"
    assert said.diagnostic_context("ready").settle == "ready"


def test_runtime_context_deadline_winner_cannot_become_late_ready(monkeypatch):
    monkeypatch.setattr(smoke_test.threading, "Thread", Mock())
    said = ORIGINAL_SAID(
        SimpleNamespace(stdout=io.StringIO("")),
        time.monotonic(),
        smoke_test.Install(False),
    )
    assert said.wait_for_ready(0) is None
    said.hear({"event": "ready", "url": "http://unused.invalid"})
    assert said.wait_for_ready(0) is None
    assert said.diagnostic_context("ready").settle == "deadline"


@pytest.mark.parametrize("ready", [False, True])
def test_runtime_context_unexpected_reader_exception_cannot_prove_eof(
    ready, monkeypatch
):
    monkeypatch.setattr(smoke_test.threading, "Thread", Mock())
    stream = Mock()
    original = RuntimeError(PRIVATE_CANARY)
    stream.readline.side_effect = original
    said = ORIGINAL_SAID(
        SimpleNamespace(stdout=stream), time.monotonic(), smoke_test.Install(False)
    )
    if ready:
        said.hear({"event": "ready", "url": "http://unused.invalid"})
    with pytest.raises(RuntimeError) as raised:
        said.read()
    assert raised.value is original
    assert said.finished.is_set() and said.evidence_lost
    assert said.diagnostic_context("ready").settle == (
        "ready" if ready else "reader_error"
    )
    assert said.diagnostic_context("stop").rejected


@pytest.mark.parametrize("operation", ["write", "flush"])
@pytest.mark.parametrize("primary", [False, True])
def test_runtime_context_late_reader_loss_and_io_failure_preserve_cleanup(
    diagnostic_runtime, monkeypatch, tmp_path, operation, primary
):
    process, _, cleanup = diagnostic_runtime
    install = smoke_test.Install(False)
    install.start_number = 1
    monkeypatch.setattr(smoke_test.threading, "Thread", Mock())
    process.stdout = io.StringIO('{"event":"shutdown","event":"shutdown"}\n')

    def reader(*args):
        said = ORIGINAL_SAID(*args)
        said.hear({"event": "ready", "url": "http://unused.invalid"})
        return said

    original = TimeoutError(PRIVATE_CANARY)

    def wait(timeout):
        install.said.read()
        if primary:
            raise original
        return 0

    process.wait.side_effect = wait
    process.poll.return_value = None if primary else 0
    monkeypatch.setattr(smoke_test, "Said", reader)
    monkeypatch.setattr(smoke_test, "stop", ORIGINAL_STOP)
    sink = RuntimeDiagnosticSink(operation)
    with redirect_stdout(sink):
        if primary:
            with pytest.raises(TimeoutError) as raised:
                smoke_test.run(tmp_path, tmp_path, 1, lambda *args: [], install)
            assert raised.value is original
        else:
            failures = smoke_test.run(tmp_path, tmp_path, 1, lambda *args: [], install)
            assert smoke_test.RUNTIME_EVENT_LOSS in failures
    assert install.said.evidence_lost
    assert install.said.diagnostic_context("ready").rejected is False
    assert install.said.diagnostic_context("stop").rejected is True
    assert sink.failures
    assert process.kill.call_count == int(primary)
    cleanup.assert_called_once()


def test_runtime_context_plain_stop_requires_drain_within_existing_budget(
    diagnostic_runtime, monkeypatch, tmp_path
):
    process, _, cleanup = diagnostic_runtime
    clock = [0.0]
    monkeypatch.setattr(
        smoke_test,
        "time",
        SimpleNamespace(monotonic=lambda: clock[0], time=lambda: clock[0]),
    )
    monkeypatch.setattr(smoke_test.threading, "Thread", Mock())
    install = smoke_test.Install(False)

    def reader(*args):
        said = ORIGINAL_SAID(*args)
        said.hear({"event": "ready", "url": "http://unused.invalid"})
        said.finished = SimpleNamespace(wait=Mock(return_value=False))
        return said

    def wait(timeout):
        clock[0] = 29
        return 0

    process.wait.side_effect = wait
    monkeypatch.setattr(smoke_test, "Said", reader)
    monkeypatch.setattr(smoke_test, "stop", ORIGINAL_STOP)
    failures = smoke_test.run(tmp_path, tmp_path, 600, lambda *args: [], install)
    assert failures == ["the runtime's shutdown events did not reach EOF"]
    process.wait.assert_called_once_with(30)
    install.said.finished.wait.assert_called_once_with(1)
    cleanup.assert_called_once()


def test_runtime_context_four_starts_keep_real_reader_completion(
    diagnostic_smoke, diagnostic_runtime, monkeypatch, capsys
):
    process, _, cleanup = diagnostic_runtime
    monkeypatch.setattr(smoke_test.threading, "Thread", InlineRuntimeReader)

    def reader(*args):
        process.stdout = io.StringIO(
            json.dumps({"event": "ready", "url": "http://unused.invalid"}) + "\n"
        )
        return ORIGINAL_SAID(*args)

    monkeypatch.setattr(smoke_test, "Said", reader)
    monkeypatch.setattr(smoke_test, "checked", lambda *args: [])
    assert smoke_test.smoke(*diagnostic_smoke) == 0
    events = safe_events(capsys)
    assert events[-1]["status"] == "passed"
    assert cleanup.call_count == 4


def test_runtime_context_optional_context_rejection_is_not_lost_history(monkeypatch):
    monkeypatch.setattr(smoke_test.threading, "Thread", InlineRuntimeReader)
    event = {"event": "timing", "phase": "database", "seconds": True}
    process = SimpleNamespace(stdout=io.StringIO(json.dumps(event) + "\n"))
    said = ORIGINAL_SAID(process, time.monotonic(), smoke_test.Install(False))
    assert said.context.rejected
    assert said.events == [event]
    assert said.finished.is_set() and not said.evidence_lost


@pytest.mark.parametrize(
    "cause", ["ready", "fatal_event", "stdout_eof", "reader_error", "deadline"]
)
def test_runtime_context_distinguishes_how_readiness_wait_ended(cause, monkeypatch):
    if cause == "reader_error":
        stream = Mock()
        stream.readline.side_effect = UnicodeDecodeError(
            "utf-8", b"\xff", 0, 1, PRIVATE_CANARY
        )
    else:
        event = (
            {"event": "ready", "url": PRIVATE_CANARY}
            if cause == "ready"
            else {
                "event": "error",
                "message": "the database did not start",
                "fatal": True,
            }
        )
        stream = io.StringIO(
            json.dumps(event) + "\n" if cause in {"ready", "fatal_event"} else ""
        )
    if cause == "deadline":
        monkeypatch.setattr(smoke_test.threading, "Thread", Mock())
    said = smoke_test.Said(
        SimpleNamespace(stdout=stream), time.monotonic(), smoke_test.Install(False)
    )
    if cause != "deadline":
        assert said.finished.wait(2)
    assert bool(said.wait_for_ready(0)) == (cause == "ready")
    record = said.context.record(
        1, "ready", smoke_test.StopObservation(), elapsed=12.345, runtime_exit=None
    )
    assert record["settle"] == cause
    assert record["elapsed_ms"] == 12345
    assert record["runtime_exit"] is None
    assert PRIVATE_CANARY not in json.dumps(record)


def test_runtime_context_completed_phases_survive_parallel_progress():
    context = smoke_test.RuntimeContext()
    for name in ("database", "queue"):
        context.observe({"event": "timing", "phase": name, "seconds": 1})
    context.observe({"event": "progress", "step": "loading", "message": PRIVATE_CANARY})
    record = context.record(1, "ready", smoke_test.StopObservation())
    assert record["phase"] == "queue"
    assert record["progress"] == "loading"
    assert record["completed_phases"] == 2 | 8
    context.observe(
        {
            "event": "timing",
            "phase": "total",
            "seconds": 2,
            "phases": {"config": 0.1, "frontend": 1},
        }
    )
    assert (
        context.record(1, "ready", smoke_test.StopObservation())["completed_phases"]
        == 1 | 2 | 8 | 64
    )


def test_runtime_context_late_reader_cannot_rewrite_readiness_deadline(monkeypatch):
    monkeypatch.setattr(smoke_test.threading, "Thread", Mock())
    event = {"event": "error", "fatal": True, "message": PRIVATE_CANARY}
    process = SimpleNamespace(stdout=io.StringIO(json.dumps(event) + "\n"))
    said = smoke_test.Said(process, time.monotonic(), smoke_test.Install(False))
    assert said.wait_for_ready(0) is None
    said.read()
    assert said.context.settle == "deadline"


def test_runtime_context_ready_with_extra_fields_is_not_trusted():
    event = {"event": "ready", "url": PRIVATE_CANARY, "extra": PRIVATE_CANARY}
    said = smoke_test.Said(
        SimpleNamespace(stdout=io.StringIO(json.dumps(event) + "\n")),
        time.monotonic(),
        smoke_test.Install(False),
    )
    assert said.finished.wait(2)
    record = said.context.record(1, "ready", smoke_test.StopObservation())
    assert record["events_rejected"]
    assert record["settle"] == "unknown"
    assert PRIVATE_CANARY not in json.dumps(record)


@pytest.mark.parametrize("elapsed", [-1, True, float("nan"), float("inf"), 3601])
def test_runtime_context_elapsed_and_exit_are_bounded(elapsed):
    record = smoke_test.RuntimeContext().record(
        1, "ready", smoke_test.StopObservation(), elapsed=elapsed, runtime_exit=True
    )
    assert type(record["elapsed_ms"]) is int and 0 <= record["elapsed_ms"] <= 3600000
    assert record["runtime_exit"] is None


def test_runtime_context_snapshot_precedes_stop_and_cleanup(
    diagnostic_runtime, tmp_path, monkeypatch, capsys
):
    now = [10.0]
    monkeypatch.setattr(smoke_test.time, "monotonic", lambda: now[0])
    _process, said, _ = diagnostic_runtime

    def wait(_):
        now[0] = 12.5

    def stop(*args):
        now[0] = 42.5
        return ["stop failed"]

    def cleanup(*args):
        now[0] = 100
        return []

    said.wait_for_ready.side_effect = wait
    monkeypatch.setattr(smoke_test, "stop", stop)
    monkeypatch.setattr(smoke_test, "left_running", cleanup)
    install = smoke_test.Install(False)
    install.start_number = 1
    assert smoke_test.run(tmp_path, tmp_path, 600, lambda *args: [], install)
    records = [
        json.loads(line[len(RUNTIME_PREFIX) :])
        for line in capsys.readouterr().out.splitlines()
        if line.startswith(RUNTIME_PREFIX)
    ]
    assert [r["elapsed_ms"] for r in records] == [2500, 32500]
    assert records[0]["runtime_exit"] == 0


@pytest.mark.parametrize(
    "message,category",
    [
        (
            "This AutoGPT version cannot open this database: it lacks or changes applied migrations (private). Install the version last used with this data, or restore a complete backup made before upgrading. The database has not been migrated.",
            "migration_incompatible",
        ),
        (
            "A database update did not finish (private). This usually means AutoGPT was closed while it was starting for the first time. If you have no data yet, delete the AutoGPT data folder and start again.",
            "migration_interrupted",
        ),
        (
            "Your AutoGPT data was created with PostgreSQL 16, and this version of AutoGPT includes PostgreSQL 17, which cannot open it. Install the AutoGPT version you used before to keep your data, or move private away to start again without it.",
            "postgres_incompatible",
        ),
        (
            "AutoGPT cannot run its database tools: this system has no private.",
            "database_tools_library",
        ),
        ("Command 'private' timed out after 600 seconds", "tool_timeout"),
        ("Command 'private' returned non-zero exit status 3.", "tool_failed"),
        ("could not set the stop event workers", "stop_event"),
        (
            "Could not settle stopped runs safely; restart AutoGPT to retry.",
            "settlement_failed",
        ),
    ],
)
def test_runtime_context_additional_source_defined_failures(message, category):
    assert smoke_test.runtime_failure(message)["error"] == category


def safe_events(capsys):
    events = [
        json.loads(line.removeprefix(SAFE_PREFIX))
        for line in capsys.readouterr().out.splitlines()
        if line.startswith(SAFE_PREFIX)
    ]
    assert PRIVATE_CANARY not in json.dumps(events)
    assert all(
        set(event)
        == {
            "version",
            "stage",
            "status",
            "reason",
            "exception",
            "contract",
            "child_exit",
            "failures",
            "start",
        }
        for event in events
    )
    return events


def contract_event(**changes):
    return {
        "version": 1,
        "stage": "contract_check",
        "status": "failed",
        "reason": "assertion_failed",
        "exception": "none",
        "contract": "nobody_is_connected",
        "child_exit": None,
        "failures": 1,
        "start": 0,
        **changes,
    }


@pytest.fixture
def diagnostic_smoke(monkeypatch, tmp_path):
    monkeypatch.setattr(smoke_test, "configure_smoke", lambda _: None)
    monkeypatch.setattr(smoke_test, "left_by_earlier_runs", lambda _: [])
    monkeypatch.setattr(smoke_test, "snapshot", lambda _: {})
    monkeypatch.setattr(smoke_test, "print_logs", lambda _: None)
    monkeypatch.setattr(smoke_test, "backend_contract", lambda _: [])
    args = SimpleNamespace(
        profile=None, quick=False, timeout=600, restart_budget=120, read_only=False
    )
    return tmp_path / "bundle", tmp_path / "data", args


@pytest.fixture
def diagnostic_runtime(monkeypatch):
    process = Mock(pid=123)
    process.poll.return_value = 0
    monkeypatch.setattr(smoke_test.subprocess, "Popen", Mock(return_value=process))
    monkeypatch.setattr(smoke_test, "shell_command", lambda _: ["unused"])
    monkeypatch.setattr(smoke_test.psutil, "Process", lambda _: object())
    monkeypatch.setattr(smoke_test, "Family", lambda *args: object())
    said = Mock()
    said.evidence_lost = False
    said.finished.wait.return_value = True
    said.wait_for_ready.return_value = "http://unused.invalid"
    monkeypatch.setattr(smoke_test, "Said", lambda *args: said)
    monkeypatch.setattr(smoke_test, "stop", lambda *args: [])
    monkeypatch.setattr(smoke_test, "stopped_gracefully", lambda *args: [])
    leftovers = Mock(return_value=[])
    monkeypatch.setattr(smoke_test, "left_running", leftovers)
    return process, said, leftovers


def test_safe_diagnostic_four_start_success_keeps_strict_sequence(
    diagnostic_smoke, diagnostic_runtime, monkeypatch, capsys
):
    monkeypatch.setattr(smoke_test, "checked", lambda *args: [])
    assert smoke_test.smoke(*diagnostic_smoke) == 0
    events = safe_events(capsys)
    assert events[-1]["stage"] == "smoke_result"
    assert events[-1]["status"] == "passed"
    launches = [e for e in events if e["stage"] == "runtime_launch"]
    assert [(e["start"], e["status"]) for e in launches] == [
        (n, status) for n in range(1, 5) for status in ("started", "passed")
    ]
    assert len(events) <= 128
    assert diagnostic_runtime[2].call_count == 4


def test_safe_diagnostic_contract_failure_skips_every_launch(
    diagnostic_smoke, monkeypatch, capsys
):
    monkeypatch.setattr(smoke_test, "backend_contract", lambda _: [PRIVATE_CANARY])
    launch = Mock(side_effect=AssertionError("must not launch"))
    monkeypatch.setattr(smoke_test, "run", launch)
    assert smoke_test.smoke(*diagnostic_smoke) == 1
    events = safe_events(capsys)
    assert events[-1]["stage"] == "smoke_result"
    assert events[-1]["status"] == "failed"
    assert events[-1]["failures"] == 1
    launch.assert_not_called()


def test_safe_diagnostic_setup_exception_is_classified_and_reraised(
    diagnostic_smoke, monkeypatch, capsys
):
    error = PermissionError(PRIVATE_CANARY)
    monkeypatch.setattr(smoke_test, "configure_smoke", Mock(side_effect=error))
    with pytest.raises(PermissionError) as raised:
        smoke_test.smoke(*diagnostic_smoke)
    assert raised.value is error
    events = safe_events(capsys)
    assert [(e["stage"], e["exception"]) for e in events if e["status"] == "error"] == [
        ("smoke_setup", "os_error"),
        ("smoke_result", "os_error"),
    ]


def test_safe_diagnostic_launch_exception_identifies_first_start(
    diagnostic_smoke, diagnostic_runtime, monkeypatch, capsys
):
    error = FileNotFoundError(PRIVATE_CANARY)
    monkeypatch.setattr(smoke_test.subprocess, "Popen", Mock(side_effect=error))
    with pytest.raises(FileNotFoundError) as raised:
        smoke_test.smoke(*diagnostic_smoke)
    assert raised.value is error
    events = safe_events(capsys)
    failure = next(
        e for e in events if e["stage"] == "runtime_launch" and e["status"] == "error"
    )
    assert failure["start"] == 1
    assert failure["exception"] == "os_error"
    assert not any(e["stage"] == "runtime_ready" for e in events)


def test_safe_diagnostic_never_ready_keeps_failure_and_cleanup(
    diagnostic_smoke, diagnostic_runtime, capsys
):
    process, said, leftovers = diagnostic_runtime
    said.wait_for_ready.return_value = None
    process.poll.return_value = 7
    assert smoke_test.smoke(*diagnostic_smoke) == 1
    events = safe_events(capsys)
    failure = next(
        e for e in events if e["stage"] == "runtime_ready" and e["status"] == "failed"
    )
    assert failure["reason"] == "not_ready"
    assert failure["child_exit"] == 7
    assert not any(e["stage"] == "runtime_checks" for e in events)
    leftovers.assert_called_once()


def test_safe_diagnostic_check_exception_still_stops_and_cleans(
    diagnostic_runtime, tmp_path, capsys
):
    def broken(*args):
        raise ValueError(PRIVATE_CANARY)

    install = smoke_test.Install(False)
    install.start_number = 1
    failures = smoke_test.run(tmp_path, tmp_path, 600, broken, install)
    assert PRIVATE_CANARY in failures[0]
    events = safe_events(capsys)
    failure = next(
        e for e in events if e["stage"] == "runtime_checks" and e["status"] == "error"
    )
    assert failure["exception"] == "value_error"
    assert events[-1]["stage"] == "runtime_cleanup"
    assert events[-1]["status"] == "passed"
    diagnostic_runtime[2].assert_called_once()


@pytest.mark.parametrize("exit_code", [0, 1, -9, 3221225477])
def test_safe_diagnostic_preflight_emits_safe_child_outcome(
    exit_code, monkeypatch, tmp_path, capsys
):
    child = contract_event(status="passed" if exit_code == 0 else "failed")
    monkeypatch.setattr(smoke_test, "shell_command", lambda _: ["unused"])
    monkeypatch.setattr(smoke_test, "contract_environment", lambda _: {})
    command = Mock(
        return_value=SimpleNamespace(
            returncode=exit_code,
            stdout=SAFE_PREFIX + json.dumps(child) + "\nBROKEN: " + PRIVATE_CANARY,
            stderr=PRIVATE_CANARY,
        )
    )
    monkeypatch.setattr(smoke_test.subprocess, "run", command)
    failures = smoke_test.backend_contract(tmp_path)
    assert bool(failures) is (exit_code != 0)
    events = safe_events(capsys)
    assert child in events
    assert events[-1]["stage"] == "backend_contract"
    assert events[-1]["child_exit"] == exit_code
    assert events[-1]["status"] == ("passed" if exit_code == 0 else "failed")
    assert command.call_args.kwargs["timeout"] == 180


@pytest.mark.parametrize(
    "error,category",
    [
        (subprocess.TimeoutExpired(PRIVATE_CANARY, 180), "timeout"),
        (ModuleNotFoundError(PRIVATE_CANARY), "module_not_found"),
        (RuntimeError(PRIVATE_CANARY), "other"),
        (KeyboardInterrupt(PRIVATE_CANARY), "keyboard_interrupt"),
    ],
)
def test_safe_diagnostic_preflight_exceptions_preserve_identity(
    error, category, monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(smoke_test, "shell_command", lambda _: ["unused"])
    monkeypatch.setattr(smoke_test, "contract_environment", lambda _: {})
    monkeypatch.setattr(smoke_test.subprocess, "run", Mock(side_effect=error))
    with pytest.raises(type(error)) as raised:
        smoke_test.backend_contract(tmp_path)
    assert raised.value is error
    events = safe_events(capsys)
    assert events[-1]["status"] == "error"
    assert events[-1]["exception"] == category


@pytest.mark.parametrize(
    "changes",
    [
        {"contract": PRIVATE_CANARY},
        {"extra": PRIVATE_CANARY},
        {"stage": "runtime_launch"},
        {"version": True},
        {"failures": True},
        {"failures": 65536},
        {"child_exit": True},
        {"start": 1},
        {"exception": PRIVATE_CANARY},
        {"child_exit": 4294967296},
    ],
)
def test_safe_diagnostic_child_forwarding_rejects_unknown_or_invalid_fields(
    changes, capsys
):
    smoke_test.forward_contract_diagnostics(
        SAFE_PREFIX + json.dumps(contract_event(**changes))
    )
    assert safe_events(capsys) == []


def test_safe_diagnostic_child_forwarding_rejects_duplicate_keys(capsys):
    payload = json.dumps(contract_event())[:-1] + ', "contract": "collect"}'
    smoke_test.forward_contract_diagnostics(SAFE_PREFIX + payload)
    assert safe_events(capsys) == []


def test_safe_diagnostic_emitter_has_closed_values_and_bounded_numbers(capsys):
    smoke_test.emit_diagnostic(
        PRIVATE_CANARY,
        PRIVATE_CANARY,
        reason=PRIVATE_CANARY,
        exception=PRIVATE_CANARY,
        contract=PRIVATE_CANARY,
        child_exit=2**64,
        failures=2**64,
        start=-3,
    )
    (event,) = safe_events(capsys)
    assert event == {
        "version": 1,
        "stage": "smoke_driver",
        "status": "error",
        "reason": "exception",
        "exception": "other",
        "contract": "unknown",
        "child_exit": None,
        "failures": 65535,
        "start": 0,
    }


def test_safe_diagnostic_early_read_only_refusal_keeps_exit_two(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["smoke_test.py", str(DESKTOP), "--read-only"])
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(smoke_test, "cannot_run_read_only", lambda: PRIVATE_CANARY)
    assert smoke_test.main() == 2
    events = safe_events(capsys)
    assert events[-1]["stage"] == "smoke_driver"
    assert events[-1]["status"] == "failed"
    assert events[-1]["reason"] == "unsupported"


def test_safe_diagnostic_timeout_forwards_only_valid_partial_child_events(
    monkeypatch, tmp_path, capsys
):
    child = contract_event(
        stage="contract_collect", status="started", contract="collect", failures=0
    )
    partial = (PRIVATE_CANARY + "\n" + SAFE_PREFIX + json.dumps(child) + "\n").encode()
    error = subprocess.TimeoutExpired(PRIVATE_CANARY, 180, output=partial)
    monkeypatch.setattr(smoke_test, "shell_command", lambda _: ["unused"])
    monkeypatch.setattr(smoke_test, "contract_environment", lambda _: {})
    monkeypatch.setattr(smoke_test.subprocess, "run", Mock(side_effect=error))
    with pytest.raises(subprocess.TimeoutExpired) as raised:
        smoke_test.backend_contract(tmp_path)
    assert raised.value is error
    events = safe_events(capsys)
    assert child in events
    assert events[-1]["reason"] == "timeout"


@pytest.mark.parametrize(
    "operation,stage", [("stop", "runtime_stop"), ("left_running", "runtime_cleanup")]
)
def test_safe_diagnostic_stop_and_cleanup_exceptions_preserve_identity(
    operation, stage, diagnostic_runtime, monkeypatch, tmp_path, capsys
):
    error = OSError(PRIVATE_CANARY)
    monkeypatch.setattr(smoke_test, operation, Mock(side_effect=error))
    with pytest.raises(OSError) as raised:
        smoke_test.run(
            tmp_path, tmp_path, 600, lambda *args: [], smoke_test.Install(False)
        )
    assert raised.value is error
    events = safe_events(capsys)
    assert any(
        e["stage"] == stage and e["status"] == "error" and e["exception"] == "os_error"
        for e in events
    )
    if operation == "stop":
        diagnostic_runtime[2].assert_called_once()


def test_safe_diagnostic_main_cleanup_error_cannot_report_driver_pass(
    monkeypatch, capsys
):
    error = OSError(PRIVATE_CANARY)
    monkeypatch.setattr(
        sys, "argv", ["smoke_test.py", str(DESKTOP), "--data-dir", str(DESKTOP)]
    )
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(smoke_test, "end_on_signals", lambda: None)
    monkeypatch.setattr(smoke_test, "smoke", lambda *args: 0)
    monkeypatch.setattr(
        smoke_test,
        "Aliases",
        lambda *args: SimpleNamespace(remove_new=Mock(side_effect=error)),
    )
    with pytest.raises(OSError) as raised:
        smoke_test.main()
    assert raised.value is error
    events = safe_events(capsys)
    assert events[-1]["stage"] == "smoke_driver"
    assert events[-1]["status"] == "error"


def test_safe_diagnostic_child_forwarding_bounds_events_and_scan(capsys):
    valid = SAFE_PREFIX + json.dumps(contract_event()) + "\n"
    smoke_test.forward_contract_diagnostics(valid * 70)
    assert len(safe_events(capsys)) == 64
    smoke_test.forward_contract_diagnostics("x" * 262144 + "\n" + valid)
    assert safe_events(capsys) == []
    smoke_test.forward_contract_diagnostics("noise\n" * 1024 + valid)
    assert safe_events(capsys) == []


@pytest.mark.parametrize("ok", [False, True])
def test_safe_diagnostic_contract_script_events_survive_parent_preflight(
    ok, monkeypatch, tmp_path, capsys
):
    spec = importlib.util.spec_from_file_location(
        "diagnostic_contract_integration", smoke_test.CONTRACT_SCRIPT
    )
    assert spec and spec.loader
    contract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contract)

    def collect(*args):
        contract.expect(ok, PRIVATE_CANARY, PRIVATE_CANARY)
        return {"fake": object()} if ok else {}

    monkeypatch.setattr(contract, "collect", collect)
    monkeypatch.setattr(contract, "CHECKS", ())
    monkeypatch.setattr(sys, "argv", ["backend_contract.py", str(tmp_path)])
    output = io.StringIO()
    with redirect_stdout(output):
        code = contract.main()
    monkeypatch.setattr(smoke_test, "shell_command", lambda _: ["unused"])
    monkeypatch.setattr(smoke_test, "contract_environment", lambda _: {})
    monkeypatch.setattr(
        smoke_test.subprocess,
        "run",
        Mock(return_value=subprocess.CompletedProcess([], code, output.getvalue(), "")),
    )
    failures = smoke_test.backend_contract(tmp_path)
    assert bool(failures) is not ok
    events = safe_events(capsys)
    assert [(e["stage"], e["status"]) for e in events] == [
        ("backend_contract", "started"),
        ("contract_collect", "started"),
        ("contract_collect", "passed" if ok else "failed"),
        ("backend_contract", "passed" if ok else "failed"),
    ]


class DiagnosticSink(io.StringIO):
    def __init__(self, stage, status, operation):
        super().__init__()
        self.stage, self.status, self.operation = stage, status, operation
        self.target = False
        self.failures = 0

    def fail_once(self):
        if self.target and not self.failures:
            self.failures += 1
            raise OSError(28, PRIVATE_CANARY)

    def write(self, text):
        if text.startswith(SAFE_PREFIX):
            event = json.loads(text[len(SAFE_PREFIX) :])
            self.target = (event["stage"], event["status"]) == (self.stage, self.status)
            if self.operation == "write":
                self.fail_once()
        return super().write(text)

    def flush(self):
        if self.operation == "flush":
            self.fail_once()


@pytest.mark.parametrize("operation", ["write", "flush"])
@pytest.mark.parametrize(
    "stage,status",
    [
        ("runtime_cleanup", "started"),
        ("runtime_launch", "passed"),
        ("runtime_ready", "error"),
    ],
)
def test_diagnostic_io_preserves_primary_and_required_cleanup(
    diagnostic_runtime, tmp_path, operation, stage, status
):
    process, said, leftovers = diagnostic_runtime
    process.poll.return_value = None
    original = TimeoutError(PRIVATE_CANARY)
    said.wait_for_ready.side_effect = original
    sink = DiagnosticSink(stage, status, operation)
    with redirect_stdout(sink), pytest.raises(TimeoutError) as raised:
        smoke_test.run(
            tmp_path, tmp_path, 600, lambda *args: [], smoke_test.Install(False)
        )
    assert raised.value is original
    assert sink.failures == 1
    process.kill.assert_called_once()
    leftovers.assert_called_once()


@pytest.mark.parametrize("operation", ["write", "flush"])
@pytest.mark.parametrize(
    "stage,status",
    [
        ("runtime_launch", "passed"),
        ("runtime_ready", "passed"),
        ("runtime_checks", "passed"),
        ("runtime_stop", "passed"),
        ("runtime_cleanup", "started"),
        ("runtime_cleanup", "passed"),
    ],
)
def test_diagnostic_io_only_failure_waits_for_cleanup_and_stays_failed(
    diagnostic_runtime, tmp_path, operation, stage, status
):
    process, _said, leftovers = diagnostic_runtime
    process.poll.return_value = None
    sink = DiagnosticSink(stage, status, operation)
    with redirect_stdout(sink):
        failures = smoke_test.run(
            tmp_path, tmp_path, 600, lambda *args: [], smoke_test.Install(False)
        )
    assert failures and PRIVATE_CANARY not in str(failures)
    assert sink.failures == 1
    process.kill.assert_called_once()
    leftovers.assert_called_once()
    with redirect_stdout(io.StringIO()):
        assert (
            smoke_test.run(
                tmp_path, tmp_path, 600, lambda *args: [], smoke_test.Install(False)
            )
            == []
        )


@pytest.mark.parametrize("operation", ["write", "flush"])
@pytest.mark.parametrize(
    "stage,status",
    [
        ("smoke_result", "started"),
        ("smoke_result", "passed"),
        ("smoke_setup", "passed"),
        ("smoke_verification", "passed"),
    ],
)
def test_diagnostic_io_smoke_cannot_return_success_after_lost_event(
    diagnostic_smoke, monkeypatch, operation, stage, status
):
    monkeypatch.setattr(smoke_test, "run", lambda *args: [])
    sink = DiagnosticSink(stage, status, operation)
    with redirect_stdout(sink):
        assert smoke_test.smoke(*diagnostic_smoke) == 1
    assert sink.failures == 1
    assert "\nOK in " not in sink.getvalue()
    with redirect_stdout(io.StringIO()):
        assert smoke_test.smoke(*diagnostic_smoke) == 0


@pytest.mark.parametrize("operation", ["write", "flush"])
@pytest.mark.parametrize("original_code", [0, 1, 2])
def test_diagnostic_io_main_checks_latch_after_its_final_event(
    monkeypatch, operation, original_code
):
    monkeypatch.setattr(
        sys, "argv", ["smoke_test.py", str(DESKTOP), "--data-dir", str(DESKTOP)]
    )
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(smoke_test, "end_on_signals", lambda: None)
    cleanup = Mock()
    monkeypatch.setattr(
        smoke_test, "Aliases", lambda *args: SimpleNamespace(remove_new=cleanup)
    )
    monkeypatch.setattr(smoke_test, "smoke", lambda *args: original_code)
    status = "passed" if original_code == 0 else "failed"
    sink = DiagnosticSink("smoke_driver", status, operation)
    with redirect_stdout(sink):
        assert smoke_test.main() == (original_code or 1)
    assert sink.failures == 1
    cleanup.assert_called_once()
    with redirect_stdout(io.StringIO()):
        assert smoke_test.main() == original_code


@pytest.mark.parametrize("operation", ["write", "flush"])
@pytest.mark.parametrize("code", [0, 1])
def test_diagnostic_io_preflight_cannot_return_success_after_final_event(
    monkeypatch, tmp_path, operation, code
):
    monkeypatch.setattr(smoke_test, "shell_command", lambda _: ["unused"])
    monkeypatch.setattr(smoke_test, "contract_environment", lambda _: {})
    monkeypatch.setattr(
        smoke_test.subprocess,
        "run",
        Mock(return_value=subprocess.CompletedProcess([], code, "", "")),
    )
    sink = DiagnosticSink(
        "backend_contract", "passed" if code == 0 else "failed", operation
    )
    with redirect_stdout(sink):
        assert smoke_test.backend_contract(tmp_path)
    assert sink.failures == 1
    with redirect_stdout(io.StringIO()):
        assert bool(smoke_test.backend_contract(tmp_path)) is (code != 0)


@pytest.mark.parametrize(
    "original", [None, ValueError(PRIVATE_CANARY), KeyboardInterrupt(PRIVATE_CANARY)]
)
def test_diagnostic_io_closed_real_stream_preserves_runtime_exceptions(
    diagnostic_runtime, tmp_path, original
):
    process, said, leftovers = diagnostic_runtime
    process.poll.return_value = None
    if original is not None:
        said.wait_for_ready.side_effect = original
    sink = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
    sink.close()
    with redirect_stdout(sink):
        if original is None:
            assert smoke_test.run(
                tmp_path, tmp_path, 600, lambda *args: [], smoke_test.Install(False)
            )
        else:
            with pytest.raises(type(original)) as raised:
                smoke_test.run(
                    tmp_path, tmp_path, 600, lambda *args: [], smoke_test.Install(False)
                )
            assert raised.value is original
    process.kill.assert_called_once()
    leftovers.assert_called_once()


def test_diagnostic_io_does_not_hide_json_construction_value_errors(monkeypatch):
    error = ValueError(PRIVATE_CANARY)
    monkeypatch.setattr(smoke_test.json, "dumps", Mock(side_effect=error))
    with pytest.raises(ValueError) as raised:
        smoke_test.emit_diagnostic("smoke_setup", "started")
    assert raised.value is error


def test_lifecycle_smoke_configures_local_catalog_before_any_child(
    tmp_path, monkeypatch
):
    from autogpt_desktop import settings
    from autogpt_desktop.layout import DataDir

    data = tmp_path / "smoke data"
    called = []

    def contract(runtime):
        user = settings.read_user_settings(DataDir(data))
        assert user.items() >= smoke_test.NO_WAY_OUT.items()
        assert user["AUTOGPT_DESKTOP_PROFILE"] == "compact"
        catalog = Path(user["SKILLS_CATALOG_PATH"])
        assert catalog.is_absolute() and catalog.is_relative_to(data)
        release = json.loads((catalog / "release.json").read_text())
        assert [p["slug"] for p in release["packages"]] == ["smoke-skill"]
        assert release["experts"] == []
        assert (
            release["catalog_sha256"]
            == "09626479caa3d7fc5ae84bb5eb9fe153b7f7c079f532841dd0c9cf4bad173cf6"
        )
        assert release["packages"][0]["files"] == [
            {
                "path": "SKILL.md",
                "executable": False,
                "sha256": "aaa16bbeaf15f80f2ac2539752df87432010fd75e42b20bda7550755f0aee498",
            }
        ]
        assert release["packages"][0]["tree_sha256"] == (
            "5dcb29319a372ab8b0db67fc4f88c86bb9679e0e47ffac63ff367d398fbf0e78"
        )
        called.append("contract")
        return []

    monkeypatch.setattr(smoke_test, "backend_contract", contract)
    monkeypatch.setattr(smoke_test, "snapshot", lambda _: {})
    monkeypatch.setattr(smoke_test, "left_by_earlier_runs", lambda _: [])
    monkeypatch.setattr(
        smoke_test, "run", lambda *args: called.append(args[3].__name__) or []
    )
    args = SimpleNamespace(
        profile="compact", quick=False, timeout=600, restart_budget=120, read_only=False
    )
    assert smoke_test.smoke(tmp_path / "bundle", data, args) == 0
    assert called == [
        "contract",
        "fresh_install",
        "restart",
        "restart_with_a_password_reset",
        "restart_after_runs_were_stopped",
    ]


def test_lifecycle_smoke_requires_catalog_publication(tmp_path, monkeypatch):
    install = smoke_test.Install(False)
    install.runtime = SimpleNamespace(children=list)
    monkeypatch.setattr(smoke_test, "wait_for", lambda *args: False)
    report = smoke_test.Report()
    smoke_test.skills_catalog(tmp_path, install, report)
    assert any("was published" in failure for failure in report.failures)


def test_lifecycle_smoke_overlay_survives_service_environment_filtering(
    tmp_path, monkeypatch
):
    from autogpt_desktop import apps, ports, settings
    from autogpt_desktop.layout import Bundle, DataDir
    from autogpt_desktop.process import base_env

    data, bundle = DataDir(tmp_path / "smoke data"), Bundle(tmp_path / "bundle")
    smoke_test.configure_smoke(data.root)
    user = settings.read_user_settings(data)
    port = {name: 20000 + index for index, name in enumerate(ports.PORT_NAMES)}
    secret = dict.fromkeys(
        (
            "POSTGRES_PASSWORD",
            "BETTER_AUTH_SECRET",
            "ENCRYPTION_KEY",
            "INTERNAL_API_KEY",
            "UNSUBSCRIBE_SECRET_KEY",
            "VAPID_PRIVATE_KEY",
            "VAPID_PUBLIC_KEY",
            "REDIS_PASSWORD",
            "RABBITMQ_DEFAULT_USER",
            "RABBITMQ_DEFAULT_PASS",
            "AUTOGPT_FRONTEND_DB_PASSWORD",
        ),
        "synthetic-smoke-only",
    )
    backend = settings.backend_environment(bundle, data, port, secret, user)
    monkeypatch.setattr(apps, "StopEvent", SimpleNamespace(create=lambda _: None))
    host = apps.host_process(bundle, data, backend, apps.MERGED[0], None)
    for env in (backend, host.env, {**base_env(), **backend}):
        assert env.items() >= smoke_test.NO_WAY_OUT.items()
        assert env["SKILLS_CATALOG_PATH"] == str(data.root / "smoke-catalog")
    frontend = settings.frontend_environment(backend, port, secret, data)
    assert not set(smoke_test.NO_WAY_OUT) & frontend.keys()


def test_lifecycle_smoke_contract_child_has_disposable_clean_environment(
    tmp_path, monkeypatch
):
    contamination = (
        "CHAT_USE_LOCAL",
        "OPENAI_API_KEY",
        "PYTHONPATH",
        "PYTHONHOME",
        "NODE_OPTIONS",
        "AUTOGPT_DESKTOP_NODE",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "AWS_PROFILE",
        "GITHUB_TOKEN",
    )
    for name in contamination:
        monkeypatch.setenv(name, "ambient-sentinel")
    seen = []

    def run(argv, **kwargs):
        env = kwargs["env"]
        assert not set(contamination) & env.keys()
        assert argv[:2] == ["bundle-python", "-B"]
        assert kwargs["timeout"] == 180 and kwargs["stdin"] == subprocess.DEVNULL
        assert env.items() >= smoke_test.NO_WAY_OUT.items()
        home = Path(env["HOME"])
        assert home.is_dir() and str(home) != os.environ.get("USERPROFILE")
        assert Path(env["CODEX_HOME"]).is_relative_to(home)
        assert Path(env["CLAUDE_CONFIG_DIR"]).is_relative_to(home)
        seen.append(home)
        return subprocess.CompletedProcess(argv, 0, "1 of 1 dependencies hold\n", "")

    monkeypatch.setattr(
        smoke_test, "subprocess", SimpleNamespace(run=run, DEVNULL=subprocess.DEVNULL)
    )
    monkeypatch.setattr(smoke_test, "shell_command", lambda _: ["bundle-python"])
    assert smoke_test.backend_contract(tmp_path) == []
    assert seen and not seen[0].exists()


class SmokeClock:
    def __init__(self):
        self.seconds = 0.0

    def monotonic(self):
        return self.seconds

    def sleep(self, seconds):
        self.seconds += seconds


@pytest.fixture
def lifecycle_stop(tmp_path, monkeypatch):
    clock = SmokeClock()
    monkeypatch.setattr(smoke_test, "time", clock)
    install = smoke_test.Install(False)
    install.profile = "compact"
    install.stopping = smoke_test.Stopping(
        "graph/run", "workers", None, "workers", None, cancels=False
    )
    install.said = SimpleNamespace(events=[], finished=threading.Event())
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs/workers.log").write_text("Received cancel for run\n")
    (tmp_path / "logs/runtime.log").write_text("")
    monkeypatch.setattr(
        smoke_test,
        "RunWatcher",
        lambda *args: SimpleNamespace(last=lambda: ("TERMINATED", 0)),
    )
    monkeypatch.setattr(smoke_test, "persisted_run", lambda *args: ("TERMINATED", 0))
    monkeypatch.setattr(smoke_test, "check_stop_records", lambda *args: None)
    monkeypatch.setattr(smoke_test, "turn_in_progress", lambda *args: None)

    class Process:
        def __init__(self, phase, total, grace=120):
            self.stdin = io.StringIO()
            self.phase, self.total, self.grace = phase, total, grace
            self.killed = False
            self.waits = []
            self.phase_events = [
                {
                    "event": "shutdown",
                    "phase": "cancellation",
                    "seconds": phase,
                    "outcome": (
                        "cancelled"
                        if install.stopping.cancels
                        else "hurried" if install.stopping.hurried else "deadline"
                    ),
                }
            ]

        def wait(self, timeout):
            self.waits.append(timeout)
            if self.total - clock.seconds > timeout:
                raise subprocess.TimeoutExpired("fake-runtime", timeout)
            clock.seconds = self.total
            message = "Stopping 1 agent run"
            install.said.events.append(
                {"step": "stopping", "message": message, "grace_seconds": self.grace}
            )
            install.said.events.extend(self.phase_events)
            install.said.finished.set()
            return 0

        def kill(self):
            self.killed = True

    return install, clock, Process


@pytest.mark.parametrize(
    "hurried,phase,total",
    [(False, 20, 57), (True, 3, 40)],
)
def test_lifecycle_smoke_separates_cancel_wait_from_frozen_host_cleanup(
    tmp_path, lifecycle_stop, hurried, phase, total
):
    install, _, Process = lifecycle_stop
    install.stopping.hurried = hurried
    process = Process(phase, total)
    assert smoke_test.stop_the_runs(process, DESKTOP, tmp_path, install) == []
    assert not process.killed
    assert install.ended["graph/run"][0] == "TERMINATED"
    assert process.waits[0] <= 120 - (3 if hurried else 0)


def test_lifecycle_smoke_rejects_missing_full_cancellation_wait(
    tmp_path, lifecycle_stop
):
    install, _, Process = lifecycle_stop
    failures = smoke_test.stop_the_runs(Process(1, 38), DESKTOP, tmp_path, install)
    assert any("waited for" in failure for failure in failures)


def test_lifecycle_smoke_fourth_restart_checks_the_public_address(
    tmp_path, monkeypatch
):
    install = smoke_test.Install(False)
    install.address = "http://127.0.0.1:10000"
    monkeypatch.setattr(
        smoke_test,
        "Browser",
        lambda _: SimpleNamespace(sign_in=lambda *args: (200, {})),
    )
    for name in (
        "ended_runs_stay_ended",
        "stopped_turns_are_not_run_again",
        "run_agent",
        "counted",
        "stop_with_a_run_in_a_long_step",
    ):
        monkeypatch.setattr(smoke_test, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(smoke_test, "executor_host", lambda _: "workers")
    monkeypatch.setattr(smoke_test, "wait_for", lambda *args: True)
    failures = smoke_test.restart_after_runs_were_stopped(
        "http://127.0.0.1:10001", tmp_path, install
    )
    assert any("address" in failure for failure in failures)


@pytest.mark.parametrize("profile", ["compact", "balanced", "isolated"])
def test_lifecycle_smoke_budget_follows_host_grace_and_parallel_tiers(
    profile, monkeypatch
):
    from autogpt_desktop import apps, supervisor

    install = smoke_test.Install(False)
    install.profile = profile
    assert smoke_test.CANCEL_WAIT_SECONDS == supervisor.CANCEL_WAIT_SECONDS
    assert smoke_test.stop_runs_budget(install) == 101
    assert smoke_test.stop_runs_budget(install) < supervisor.STOPPING_RUNS_GRACE_SECONDS
    original = apps.host_stop_timeout
    monkeypatch.setattr(
        apps,
        "host_stop_timeout",
        lambda group: 38 if "executor" in group.services else original(group),
    )
    assert smoke_test.stop_runs_budget(install) == 109


@pytest.mark.parametrize(
    "phase,cancels,hurried", [(20, False, True), (20, True, False)]
)
def test_lifecycle_smoke_rejects_slow_cancel_or_ignored_hurry(
    tmp_path, lifecycle_stop, phase, cancels, hurried
):
    install, _, Process = lifecycle_stop
    install.stopping.cancels, install.stopping.hurried = cancels, hurried
    failures = smoke_test.stop_the_runs(Process(phase, 57), DESKTOP, tmp_path, install)
    assert any("wait" in failure for failure in failures)


def test_lifecycle_smoke_cancellation_can_finish_before_slow_cleanup(
    tmp_path, lifecycle_stop
):
    install, _, Process = lifecycle_stop
    install.stopping.cancels = True
    assert smoke_test.stop_the_runs(Process(2, 35), DESKTOP, tmp_path, install) == []


@pytest.mark.parametrize(
    "seconds", [True, None, "20", -1, float("nan"), float("inf"), 58]
)
def test_lifecycle_smoke_rejects_invalid_cancellation_measurements(
    tmp_path, lifecycle_stop, seconds
):
    install, _, Process = lifecycle_stop
    process = Process(20, 57)
    process.phase_events[0]["seconds"] = seconds
    failures = smoke_test.stop_the_runs(process, DESKTOP, tmp_path, install)
    assert any("cancellation phase" in failure for failure in failures)


@pytest.mark.parametrize(
    "outcome", [None, "cancelled", "hurried", "inactive", "unknown"]
)
def test_lifecycle_smoke_requires_deadline_outcome_for_frozen_run(
    tmp_path, lifecycle_stop, outcome
):
    install, _, Process = lifecycle_stop
    process = Process(20, 57)
    process.phase_events[0]["outcome"] = outcome
    failures = smoke_test.stop_the_runs(process, DESKTOP, tmp_path, install)
    assert any("waited for" in failure for failure in failures)


@pytest.mark.parametrize("count", [0, 2])
def test_lifecycle_smoke_requires_exactly_one_cancellation_event(
    tmp_path, lifecycle_stop, count
):
    install, _, Process = lifecycle_stop
    process = Process(20, 57)
    process.phase_events *= count
    failures = smoke_test.stop_the_runs(process, DESKTOP, tmp_path, install)
    assert any("cancellation phase" in failure for failure in failures)


def test_lifecycle_smoke_drains_final_events_after_process_exit(
    tmp_path, lifecycle_stop
):
    install, _, Process = lifecycle_stop
    process = Process(20, 57)
    deferred = process.phase_events
    process.phase_events = []
    waits = []

    def drained(timeout):
        waits.append(timeout)
        install.said.events.extend(deferred)
        return True

    install.said.finished = SimpleNamespace(set=lambda: None, wait=drained)
    assert smoke_test.stop_the_runs(process, DESKTOP, tmp_path, install) == []
    assert waits and 0 < waits[0] <= 3


def test_lifecycle_smoke_refuses_incomplete_event_drain(tmp_path, lifecycle_stop):
    install, _, Process = lifecycle_stop
    install.said.finished = SimpleNamespace(set=lambda: None, wait=lambda _: False)
    failures = smoke_test.stop_the_runs(Process(20, 57), DESKTOP, tmp_path, install)
    assert any("shutdown events" in failure for failure in failures)


def test_lifecycle_smoke_event_reader_reports_eof_separately_from_ready():
    install = smoke_test.Install(False)
    event = {
        "event": "shutdown",
        "phase": "cancellation",
        "seconds": 20,
        "outcome": "deadline",
    }
    process = SimpleNamespace(stdout=io.StringIO(json.dumps(event) + "\n"))
    said = smoke_test.Said(process, time.monotonic(), install)
    assert said.finished.wait(2)
    assert said.events == [event]


def test_lifecycle_smoke_still_requires_enough_advertised_grace(
    tmp_path, lifecycle_stop
):
    install, _, Process = lifecycle_stop
    failures = smoke_test.stop_the_runs(
        Process(20, 57, grace=60), DESKTOP, tmp_path, install
    )
    assert any("shell was asked" in failure for failure in failures)


@pytest.mark.parametrize("hurried", [False, True])
def test_lifecycle_smoke_total_deadline_includes_the_hurry_delay(
    tmp_path, lifecycle_stop, hurried
):
    install, _, Process = lifecycle_stop
    install.stopping.hurried = hurried
    process = Process(3 if hurried else 20, 102)
    failures = smoke_test.stop_the_runs(process, DESKTOP, tmp_path, install)
    assert process.killed and process.waits == [98 if hurried else 101]
    assert failures == [
        "the runtime did not stop within 101s of being asked to stop the runs"
    ]


@pytest.mark.parametrize("status,unfinished", [("RUNNING", 0), ("TERMINATED", 1)])
def test_lifecycle_smoke_keeps_the_persisted_termination_oracle(
    tmp_path, lifecycle_stop, monkeypatch, status, unfinished
):
    install, _, Process = lifecycle_stop
    monkeypatch.setattr(smoke_test, "persisted_run", lambda *args: (status, unfinished))
    failures = smoke_test.stop_the_runs(Process(20, 57), DESKTOP, tmp_path, install)
    assert any("persisted" in failure for failure in failures)


def test_lifecycle_smoke_keeps_leftover_failures(tmp_path, monkeypatch):
    process = SimpleNamespace(pid=123, poll=lambda: 0)

    def popen(*args, **kwargs):
        return process

    monkeypatch.setattr(
        smoke_test, "subprocess", SimpleNamespace(Popen=popen, PIPE=subprocess.PIPE)
    )
    monkeypatch.setattr(
        smoke_test, "psutil", SimpleNamespace(Process=lambda _: object())
    )
    monkeypatch.setattr(smoke_test, "Family", lambda *args: object())
    monkeypatch.setattr(
        smoke_test,
        "Said",
        lambda *args: SimpleNamespace(
            wait_for_ready=lambda _: "url",
            finished=SimpleNamespace(wait=lambda _: True),
        ),
    )
    monkeypatch.setattr(smoke_test, "shell_command", lambda _: ["bundle-python"])
    monkeypatch.setattr(smoke_test, "stop", lambda *args: [])
    monkeypatch.setattr(smoke_test, "stopped_gracefully", lambda *args: [])
    monkeypatch.setattr(
        smoke_test, "left_running", lambda _: ["owned process left running"]
    )
    failures = smoke_test.run(
        DESKTOP, tmp_path, 600, lambda *args: [], smoke_test.Install(False)
    )
    assert failures == ["owned process left running"]


posix_only = pytest.mark.skipif(
    sys.platform == "win32", reason="permission bits are POSIX's"
)
posix_only_signals = pytest.mark.skipif(
    sys.platform == "win32", reason="Windows has no hang-up signal"
)


@pytest.fixture
def runtime(tmp_path: Path) -> Path:
    root = tmp_path / "runtime"
    (root / "frontend" / ".next" / "server").mkdir(parents=True)
    (root / "frontend" / "server.js").write_text("server")
    (root / "prisma" / "node_modules" / "@prisma" / "engines").mkdir(parents=True)
    return root


def test_a_directory_a_run_creates_is_a_change_even_when_empty(runtime: Path):
    before = settled(runtime)
    assert smoke_test.changed_files(before, smoke_test.snapshot(runtime)) == []

    (runtime / "frontend" / ".next" / "cache" / "images").mkdir(parents=True)

    (failure,) = smoke_test.changed_files(before, smoke_test.snapshot(runtime))
    # The two new directories, and the one they were created in.
    assert "changed 3 file(s) or directories" in failure
    assert os.path.join("frontend", ".next", "cache") + os.sep in failure


def test_a_file_that_came_and_went_during_the_run_is_a_change(runtime: Path):
    """A temporary file, a lock, a pid file: gone by the time the run is
    over, and a write an installed bundle would have refused."""
    before = settled(runtime)
    passing = runtime / "frontend" / ".next" / "server" / "x.tmp"
    make_later_than(
        runtime / "frontend" / ".next" / "server", lambda: passing.write_text("x")
    )
    passing.unlink()

    (failure,) = smoke_test.changed_files(before, smoke_test.snapshot(runtime))
    assert "changed 1 file(s) or directories" in failure
    assert os.path.join("frontend", ".next", "server") + os.sep in failure


def test_a_file_in_the_top_of_the_bundle_that_came_and_went_is_a_change(runtime: Path):
    before = settled(runtime)
    make_later_than(runtime, lambda: (runtime / "lock").write_text("x"))
    (runtime / "lock").unlink()
    assert len(smoke_test.changed_files(before, smoke_test.snapshot(runtime))) == 1


def settled(runtime: Path) -> dict:
    """A snapshot of a bundle that was not touched a moment ago. A real one
    was built minutes before it is run; here it was made in the same tick of
    a clock that, on Windows, moves every 16 ms, and a directory changed
    again within that tick keeps the time it had."""
    past = time.time() - 600
    for directory in [runtime, *(path for path in runtime.rglob("*") if path.is_dir())]:
        if not directory.is_symlink() and not is_junction(directory):
            os.utime(directory, (past, past))
    return smoke_test.snapshot(runtime)


def is_junction(path: Path) -> bool:
    return sys.platform == "win32" and bool(
        os.lstat(path).st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT
    )


def make_later_than(directory: Path, change) -> None:
    """Make `change`, and make sure the directory's modification time is
    seen to move on a file system whose clock is coarser than this test."""
    was = directory.stat().st_mtime_ns
    change()
    if directory.stat().st_mtime_ns == was:
        os.utime(directory, ns=(was + 1_000_000, was + 1_000_000))


def test_a_link_a_run_creates_or_repoints_is_a_change(runtime: Path, tmp_path: Path):
    one, other = tmp_path / "one", tmp_path / "other"
    one.mkdir()
    other.mkdir()
    link = runtime / "frontend" / "link"
    make_link(link, one)
    before = settled(runtime)
    assert before[os.path.join("frontend", "link")][0] == "link"
    assert not any(
        name.startswith(os.path.join("frontend", "link") + os.sep) for name in before
    )

    remove_link(link)
    make_link(link, other)

    (failure,) = smoke_test.changed_files(before, smoke_test.snapshot(runtime))
    assert os.path.join("frontend", "link") in failure


@posix_only
def test_a_socket_or_a_pipe_a_run_leaves_is_a_change(runtime: Path):
    before = settled(runtime)
    os.mkfifo(runtime / "frontend" / "pipe")
    (failure,) = smoke_test.changed_files(before, smoke_test.snapshot(runtime))
    assert os.path.join("frontend", "pipe") in failure


def make_link(link: Path, target: Path) -> None:
    if sys.platform == "win32":
        from autogpt_desktop import winlinks

        winlinks.create_junction(target, link)  # a symlink needs a privilege
    else:
        link.symlink_to(target, target_is_directory=True)


def remove_link(link: Path) -> None:
    if sys.platform == "win32":
        link.rmdir()
    else:
        link.unlink()


def test_a_file_added_changed_or_removed_is_a_change(runtime: Path):
    before = settled(runtime)
    (runtime / "frontend" / "server.js").write_text("rewritten")
    (runtime / "prisma" / "libquery_engine.so.node").write_text("downloaded")
    (failure,) = smoke_test.changed_files(before, smoke_test.snapshot(runtime))
    # Both files, and the directory the new one is in.
    assert "changed 3 file(s)" in failure
    assert os.path.join("frontend", "server.js") in failure
    assert os.path.join("prisma", "libquery_engine.so.node") in failure

    again = settled(runtime)
    (runtime / "prisma" / "libquery_engine.so.node").unlink()
    (failure,) = smoke_test.changed_files(again, smoke_test.snapshot(runtime))
    assert os.path.join("prisma", "libquery_engine.so.node") in failure


def test_what_an_earlier_run_left_in_the_bundle_fails_the_run(runtime: Path, capsys):
    assert smoke_test.left_by_earlier_runs(runtime) == []

    (runtime / "frontend" / ".next" / "cache").mkdir()
    engines = runtime / "prisma" / "node_modules" / "@prisma" / "engines"
    (engines / "libquery_engine-debian-openssl-1.1.x.so.node").write_text("downloaded")

    (failure,) = smoke_test.left_by_earlier_runs(runtime)
    assert "libquery_engine-debian-openssl-1.1.x.so.node" in failure
    assert "cache" in failure and "--only seal" in failure


def test_a_log_that_speaks_of_a_refused_write_fails_a_read_only_run(tmp_path: Path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "api.log").write_text("INFO started\nINFO serving\n")
    assert smoke_test.refused_writes(tmp_path) == []

    (logs / "frontend.log").write_text(
        " ⨯ Failed to write image to cache abc Error: EACCES: permission denied, mkdir\n",
        encoding="utf-8",
    )
    (logs / "rabbitmq.log").write_text(
        "could not open: Read-only file system\nagain: EROFS\n"
    )

    failures = smoke_test.refused_writes(tmp_path)
    assert [failure.split(" ", 1)[0] for failure in failures] == [
        "frontend.log",
        "rabbitmq.log",
    ]
    assert "2 line(s)" in failures[1]

    # The backend's own file logs are a folder further down (LOG_DIR), and
    # the runtime's is beside the services'.
    (logs / "backend").mkdir()
    (logs / "backend" / "rest.log").write_text(
        "PermissionError: [Errno 13] Permission denied: 'x'\n"
    )
    (logs / "runtime.log").write_text("OSError: [Errno 30] Read-only file system\n")
    names = [
        failure.split(" ", 1)[0] for failure in smoke_test.refused_writes(tmp_path)
    ]
    assert names == ["backend/rest.log", "frontend.log", "rabbitmq.log", "runtime.log"]


def test_the_temporary_data_directory_is_removed_with_what_is_in_it(tmp_path: Path):
    data = tmp_path / "autogpt-smoke-x"
    (data / "postgres" / "base").mkdir(parents=True)
    (data / "postgres" / "PG_VERSION").write_text("18")
    (data / "rabbitmq").mkdir()
    cookie = data / "rabbitmq" / ".erlang.cookie"  # as rabbitmq.prepare writes it
    cookie.write_text("COOKIE")
    cookie.chmod(0o400)

    smoke_test.remove_data(data)

    assert not data.exists()
    smoke_test.remove_data(data)  # already gone: nothing to do, no error


def test_read_only_mode_is_refused_where_it_proves_nothing(monkeypatch):
    monkeypatch.setattr(smoke_test.sys, "platform", "win32")
    assert "macOS and Linux" in str(smoke_test.cannot_run_read_only())
    if hasattr(os, "geteuid"):
        monkeypatch.setattr(smoke_test.sys, "platform", "linux")
        monkeypatch.setattr(os, "geteuid", lambda: 0)
        assert "root" in str(smoke_test.cannot_run_read_only())
        monkeypatch.setattr(os, "geteuid", lambda: 1000)
        assert smoke_test.cannot_run_read_only() is None


@posix_only
def test_the_bundle_is_unwritable_for_the_run_and_as_it_was_afterwards(runtime: Path):
    script = runtime / "frontend" / "server.js"
    script.chmod(0o755)
    before = {
        path: stat.S_IMODE(path.stat().st_mode)
        for path in [runtime, *runtime.rglob("*")]
    }

    with (
        pytest.raises(RuntimeError, match="the run failed"),
        smoke_test.unwritable(runtime),
    ):
        assert stat.S_IMODE(script.stat().st_mode) == 0o555
        if os.geteuid() != 0:
            with pytest.raises(PermissionError):
                (runtime / "frontend" / ".next" / "cache").mkdir()
            with pytest.raises(PermissionError):
                script.write_text("changed")
        raise RuntimeError("the run failed")

    after = {
        path: stat.S_IMODE(path.stat().st_mode)
        for path in [runtime, *runtime.rglob("*")]
    }
    assert after == before


@posix_only
def test_a_run_being_ended_can_put_the_permissions_back_first_and_only_once(
    runtime: Path,
):
    script = runtime / "frontend" / "server.js"
    with smoke_test.unwritable(runtime) as give_back:
        assert not os.access(runtime / "frontend", os.W_OK) or os.geteuid() == 0
        give_back()
        assert stat.S_IMODE(script.stat().st_mode) & 0o200
        script.chmod(0o400)  # what happens after is not undone a second time
        give_back()
    assert stat.S_IMODE(script.stat().st_mode) == 0o400


@pytest.fixture
def handlers():
    """This test process's own answers to the signals, put back afterwards."""
    before = {number: signal.getsignal(number) for number in smoke_test.END_SIGNALS}
    yield before
    for number, handler in before.items():
        signal.signal(number, handler)


def test_being_told_to_end_is_an_interrupt_so_that_everything_is_cleared_up(handlers):
    """Left to Python, SIGTERM ends the process where it stands: no
    `finally` runs, the data directory stays, a read-only bundle stays
    read-only."""
    smoke_test.end_on_signals()
    for number in smoke_test.END_SIGNALS:
        if number == signal.SIGINT:
            continue
        with pytest.raises(KeyboardInterrupt, match=signal.Signals(number).name):
            signal.getsignal(number)(number, None)


@posix_only
def test_a_signal_that_arrives_while_permissions_go_back_does_not_stop_that(handlers):
    smoke_test.end_on_signals()
    with smoke_test.deaf_to_signals():
        os.kill(os.getpid(), signal.SIGTERM)  # and no KeyboardInterrupt
        assert signal.getsignal(signal.SIGTERM) is signal.SIG_IGN
    with pytest.raises(KeyboardInterrupt):  # heard again afterwards
        os.kill(os.getpid(), signal.SIGTERM)
        signal.pause()


# A stand-in runtime: starts a service that outlives it, says which, and
# stays until its stdin closes.
RUNTIME = """
import subprocess, sys
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
print(child.pid, flush=True)
sys.stdin.read()
"""


def test_only_what_the_runtime_started_counts_as_left_running(tmp_path: Path):
    """The bundle's interpreter also runs the unit tests, perhaps right now
    and for somebody else: a process is a leftover because the runtime
    started it, not because of where its executable is."""
    bystander = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    runtime = subprocess.Popen(
        [sys.executable, "-c", RUNTIME],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert runtime.stdout and runtime.stdin
        service = psutil.Process(int(runtime.stdout.readline()))
        family = smoke_test.Family(psutil.Process(runtime.pid), tmp_path, BUNDLE)
        family.look()
        runtime.stdin.close()
        runtime.wait(20)

        (failure,) = smoke_test.left_running(family)

        assert "processes left running" in failure
        assert not smoke_test.is_alive(service)  # ended, so the machine is left clean
        assert bystander.poll() is None
    finally:
        for process in (bystander, runtime):
            process.kill()
            process.wait(10)


def test_a_runtime_that_took_everything_with_it_left_nothing(tmp_path: Path):
    runtime = subprocess.Popen([sys.executable, "-c", "pass"])
    family = smoke_test.Family(psutil.Process(runtime.pid), tmp_path, BUNDLE)
    runtime.wait(20)
    assert smoke_test.left_running(family) == []


def test_a_process_that_left_the_family_before_it_was_seen_is_still_found(
    tmp_path: Path, monkeypatch
):
    """A daemon: by the time anyone looks, its parent has gone and it is
    nobody's child. It runs from the bundle, was started after the runtime,
    and inherited what the runtime was started with; the developer's tests,
    running from the same bundle, did not, and are left alone."""
    monkeypatch.setattr(smoke_test.Family, "look", lambda self: None)
    bystander = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    runtime = subprocess.Popen(
        [sys.executable, "-c", RUNTIME],
        env={**os.environ, "AUTOGPT_DESKTOP_DATA_DIR": str(tmp_path / "data")},
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    daemon = None
    try:
        assert runtime.stdout and runtime.stdin
        daemon = psutil.Process(int(runtime.stdout.readline()))
        family = smoke_test.Family(
            psutil.Process(runtime.pid), tmp_path / "data", BUNDLE
        )
        runtime.stdin.close()
        runtime.wait(20)

        (failure,) = smoke_test.left_running(family)

        assert "processes left running" in failure
        assert not smoke_test.is_alive(daemon)
        assert bystander.poll() is None
    finally:
        for process in (bystander, runtime):
            process.kill()
            process.wait(10)
        if daemon and smoke_test.is_alive(daemon):
            daemon.kill()


def test_a_process_started_with_this_runs_data_directory_is_of_this_run(
    tmp_path: Path, monkeypatch
):
    """Whoever its parent is: here, this test, which is alive and no part of
    the run."""
    monkeypatch.setattr(smoke_test.Family, "look", lambda self: None)
    runtime = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE
    )
    sleep = [sys.executable, "-c", "import time; time.sleep(120)"]
    family = smoke_test.Family(psutil.Process(runtime.pid), tmp_path / "data", BUNDLE)
    of_this_run = subprocess.Popen(
        sleep, env={**os.environ, "RABBITMQ_BASE": str(tmp_path / "data" / "rabbitmq")}
    )
    bystander = subprocess.Popen(
        sleep, env={**os.environ, "RABBITMQ_BASE": str(tmp_path / "other")}
    )
    try:
        runtime.communicate(timeout=20)

        (failure,) = smoke_test.left_running(family)

        assert "processes left running" in failure
        assert of_this_run.wait(10) is not None
        assert bystander.poll() is None
    finally:
        for process in (bystander, of_this_run, runtime):
            process.kill()
            process.wait(10)


def test_a_process_working_in_the_data_directory_is_of_this_run(tmp_path: Path):
    """Valkey is started with nothing but a working directory to tell."""
    (tmp_path / "data" / "valkey").mkdir(parents=True)
    sleep = [sys.executable, "-c", "import time; time.sleep(120)"]
    inside = subprocess.Popen(sleep, cwd=tmp_path / "data" / "valkey")
    outside = subprocess.Popen(sleep, cwd=tmp_path)
    try:
        marks = [str(tmp_path / "data")]
        assert smoke_test.mentions(psutil.Process(inside.pid), marks)
        assert not smoke_test.mentions(psutil.Process(outside.pid), marks)
    finally:
        for process in (inside, outside):
            process.kill()
            process.wait(10)


# Where this interpreter's image is, as the check sees it for the processes
# started below. Not sys.executable: a framework build on macOS (GitHub's
# runners have one) runs Python.app from another directory.
BUNDLE = Path(psutil.Process().exe()).resolve().parent


@posix_only_signals
def test_a_run_started_with_nohup_outlives_its_terminal():
    """nohup runs a program with the hang-up signal ignored. The smoke test
    turns signals that would kill it into a tidy end, and must leave that one
    alone, or the run ends when the terminal it was started from closes."""
    probe = (
        "import signal, sys; sys.path.insert(0, sys.argv[1]); import smoke_test; "
        "signal.signal(signal.SIGHUP, signal.SIG_IGN); smoke_test.end_on_signals(); "
        "print(signal.getsignal(signal.SIGHUP) is signal.SIG_IGN, "
        "callable(signal.getsignal(signal.SIGTERM)))"
    )
    answer = subprocess.run(
        [sys.executable, "-c", probe, str(DESKTOP / "build")],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert answer.stdout.split() == ["True", "True"], answer.stderr


# --- the links a run leaves outside the data directory ------------------------


@pytest.fixture
def link_roots(tmp_path: Path, monkeypatch) -> list[Path]:
    roots = [
        tmp_path / "cache" / "AutoGPT" / "links",
        tmp_path / "shared" / "AutoGPT-user",
    ]
    (tmp_path / "shared").mkdir()
    monkeypatch.setattr(smoke_test, "alias_roots", lambda: roots)
    return roots


def test_the_links_a_run_made_to_its_data_and_to_the_bundle_go_with_it(
    runtime: Path, tmp_path: Path, link_roots: list[Path]
):
    """With --data-dir too: the directory is the caller's, the link the
    runtime made to it is not."""
    data = tmp_path / "autogpt smoke"
    (data / "rabbitmq").mkdir(parents=True)
    (runtime / "erlang").mkdir()
    aliases = smoke_test.Aliases(runtime, data)

    for root in link_roots:
        root.mkdir(parents=True)
    make_link(link_roots[0] / "to-data", data / "rabbitmq")
    make_link(link_roots[0] / "to-erlang", runtime / "erlang")
    make_link(link_roots[1] / "to-data-too", data / "rabbitmq")
    aliases.remove_new()

    assert not any(os.path.lexists(root) for root in link_roots)
    assert not (
        tmp_path / "cache" / "AutoGPT"
    ).exists()  # made for the links, and empty
    assert (tmp_path / "shared").is_dir()
    assert (data / "rabbitmq").is_dir() and (runtime / "erlang").is_dir()


def test_links_that_were_there_or_lead_elsewhere_are_somebody_elses(
    runtime: Path, tmp_path: Path, link_roots: list[Path]
):
    data = tmp_path / "autogpt smoke"
    (data / "rabbitmq").mkdir(parents=True)
    installed = tmp_path / "the installed app's data"
    installed.mkdir()
    root = link_roots[0]
    root.mkdir(parents=True)
    make_link(root / "from-an-earlier-run", data / "rabbitmq")
    (root / "a-folder").mkdir()
    aliases = smoke_test.Aliases(runtime, data)

    make_link(root / "the-installed-apps", installed)  # it was started meanwhile
    make_link(root / "this-runs", data / "rabbitmq")
    aliases.remove_new()

    assert sorted(path.name for path in root.iterdir()) == [
        "a-folder",
        "from-an-earlier-run",
        "the-installed-apps",
    ]


def test_a_link_is_known_by_where_it_goes_not_by_how_the_path_is_spelled(
    tmp_path: Path,
):
    data = tmp_path / "data"
    (data / "rabbitmq").mkdir(parents=True)
    make_link(tmp_path / "another way in", tmp_path / "data")
    make_link(tmp_path / "link", tmp_path / "another way in" / "rabbitmq")

    assert smoke_test.leads_into(tmp_path / "link", data)
    assert not smoke_test.leads_into(tmp_path / "link", tmp_path / "elsewhere")


def test_a_stale_postmaster_pid_is_not_taken_for_this_runs_server(tmp_path: Path):
    """It names a process that is gone, or by now somebody else's, which
    must not be counted, let alone ended."""
    (tmp_path / "postgres").mkdir()
    (tmp_path / "postgres" / "postmaster.pid").write_text(f"{os.getppid()}\n")
    me = psutil.Process()
    assert smoke_test.nobodys_children(tmp_path, me) == []
    (tmp_path / "postgres" / "postmaster.pid").write_text("not a pid\n")
    assert smoke_test.nobodys_children(tmp_path, me) == []


# --- how long a start takes ----------------------------------------------------


def start(name: str, to_ready: float | None, **phases: float) -> "smoke_test.Start":
    return smoke_test.Start(
        name, to_ready, {"phase": "total", "phases": phases} if phases else None
    )


def test_a_restart_over_its_budget_fails_the_run_and_a_first_start_does_not():
    """A first start also creates the database; a restart is what every
    start of the app costs."""
    starts = [
        start("fresh_install", 300.0, database=40.0),
        start("restart", 95.0, queue=70.0, services=20.0),
        start("restart_with_a_password_reset", 30.0, queue=6.0),
    ]

    (failure,) = smoke_test.slow_restarts(starts, 90)

    assert "`restart` took 95s" in failure
    assert "budget is 90s (--restart-budget)" in failure
    assert "queue 70.0, services 20.0" in failure
    assert smoke_test.slow_restarts(starts, 120) == []


def test_a_start_that_never_became_ready_is_not_also_a_slow_one():
    """It is already a failure, in its own words."""
    assert smoke_test.slow_restarts([start("restart", None)], 90) == []


def test_the_table_of_starts_has_every_phase_and_a_dash_for_one_that_did_not_run(
    capsys,
):
    smoke_test.print_starts(
        [
            start("fresh_install", 24.1, config=0.03, database=6.5, queue=7.0),
            start("restart", None),
        ]
    )
    printed = capsys.readouterr().out.splitlines()
    header, first, second = printed[-3:]
    assert header.split() == ["start", "to", "ready", *smoke_test.PHASES]
    assert first.split() == [
        "fresh",
        "install",
        "24.1",
        "0.0",
        "6.5",
        "-",
        "7.0",
        "-",
        "-",
        "-",
    ]
    assert second.split()[:2] == ["restart", "never"]


def test_the_phases_the_table_shows_are_the_ones_the_runtime_reports():
    sys.path.insert(0, str(DESKTOP / "runtime"))
    timing = importlib.import_module("autogpt_desktop.timing")
    assert smoke_test.PHASES == timing.PHASES


class NoAliases:
    def __init__(self, runtime: Path, data: Path) -> None:
        pass

    def remove_new(self) -> None:
        pass


def test_the_restart_budget_is_a_parameter(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        smoke_test,
        "smoke",
        lambda runtime, data, args, give_back=None: seen.update(vars(args)) or 0,
    )
    monkeypatch.setattr(smoke_test, "end_on_signals", lambda: None)
    monkeypatch.setattr(smoke_test, "Aliases", NoAliases)
    monkeypatch.setattr(sys, "path", list(sys.path))  # main() adds the runtime to it
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "smoke_test.py",
            str(DESKTOP),
            "--restart-budget",
            "35",
            "--data-dir",
            str(DESKTOP),
        ],
    )
    assert smoke_test.main() == 0
    assert seen["restart_budget"] == 35
    monkeypatch.setattr(
        sys, "argv", ["smoke_test.py", str(DESKTOP), "--data-dir", str(DESKTOP)]
    )
    smoke_test.main()
    assert seen["restart_budget"] == smoke_test.RESTART_BUDGET_SECONDS


@pytest.mark.parametrize("record", ["stopped", "ended", "both"])
def test_a_stopped_turn_keeps_its_exact_identity_until_the_next_start(
    tmp_path: Path, record: str
):
    (tmp_path / "run").mkdir()
    stopping = smoke_test.Stopping("graph/run", "workers", "chat", "workers", None)
    if record in {"stopped", "both"}:
        (tmp_path / "run" / "stopped-runs.json").write_text(
            json.dumps(
                {
                    "graph_runs": ["run"],
                    "copilot_turns": ["chat"],
                    "copilot_turn_ids": {"chat": "turn"},
                }
            )
        )
    if record in {"ended", "both"}:
        (tmp_path / "run" / "ended-turns.json").write_text(
            json.dumps(
                [
                    {"session_id": "chat", "turn_id": "turn"},
                ]
            )
        )
    report = smoke_test.Report()
    smoke_test.check_stop_records(tmp_path, stopping, "turn", report)
    assert report.failures == []


@pytest.mark.parametrize(
    "record", [None, [], ["chat"], [{"session_id": "chat", "turn_id": "later"}]]
)
def test_missing_or_wrong_turn_intent_cannot_pass_the_stop_smoke_check(
    tmp_path: Path, record
):
    (tmp_path / "run").mkdir()
    if record is not None:
        (tmp_path / "run" / "ended-turns.json").write_text(json.dumps(record))
    stopping = smoke_test.Stopping("graph/run", "workers", "chat", "workers", None)
    report = smoke_test.Report()
    smoke_test.check_stop_records(tmp_path, stopping, "turn", report)
    assert report.failures


@pytest.mark.parametrize(
    "field,value",
    [
        ("graph_runs", ["other-run"]),
        ("copilot_turns", ["other-chat"]),
        ("copilot_turn_ids", {"chat": "later-turn"}),
    ],
)
def test_retained_stop_intent_cannot_name_different_work(
    tmp_path: Path, field: str, value
):
    (tmp_path / "run").mkdir()
    record = {
        "graph_runs": ["run"],
        "copilot_turns": ["chat"],
        "copilot_turn_ids": {"chat": "turn"},
    }
    record[field] = value
    (tmp_path / "run" / "stopped-runs.json").write_text(json.dumps(record))
    report = smoke_test.Report()
    smoke_test.check_stop_records(
        tmp_path,
        smoke_test.Stopping("graph/run", "workers", "chat", "workers", None),
        "turn",
        report,
    )
    assert report.failures


def test_persisted_run_refuses_a_cluster_that_is_still_running(tmp_path: Path):
    (tmp_path / "postgres").mkdir()
    (tmp_path / "postgres/postmaster.pid").write_text("123")
    with pytest.raises(RuntimeError, match="while PostgreSQL is running"):
        smoke_test.persisted_run(DESKTOP, tmp_path, "run")


@pytest.mark.skipif(
    not os.environ.get("AUTOGPT_TEST_POSTGRES_RUNTIME"),
    reason="requires an explicitly selected existing PostgreSQL runtime",
)
@pytest.mark.parametrize(
    "graph_status,node_status,expected",
    [
        ("TERMINATED", "FAILED", ("TERMINATED", 0)),
        ("RUNNING", "RUNNING", ("RUNNING", 1)),
    ],
)
def test_stopped_run_oracle_reads_durable_state_without_reconciliation(
    tmp_path: Path,
    graph_status: str,
    node_status: str,
    expected: tuple[str, int],
):
    from autogpt_desktop import ports, postgres
    from autogpt_desktop.layout import Bundle, DataDir
    from autogpt_desktop.process import wait_until

    bundle = Bundle(Path(os.environ["AUTOGPT_TEST_POSTGRES_RUNTIME"]))
    data = DataDir(tmp_path / "source")
    data.prepare()
    password = "synthetic-smoke-oracle-only"
    data.runtime_env.write_text(f"POSTGRES_PASSWORD={password}\n")
    postgres.initialize(bundle, data, password)
    port = ports.allocate(data.ports_file, ("postgres",))["postgres"]
    server = postgres.process(bundle, data, port)
    connection = None

    def connected():
        nonlocal connection
        try:
            connection = smoke_test.database(data.root)
            return True
        except smoke_test.psycopg2.OperationalError:
            return False

    try:
        server.start()
        assert wait_until(connected, 30, interval=0.1)
        with connection.cursor() as cursor:
            cursor.execute("CREATE SCHEMA platform")
            cursor.execute(
                'CREATE TABLE platform."AgentGraphExecution" (id text, "executionStatus" text)'
            )
            cursor.execute(
                'CREATE TABLE platform."AgentNodeExecution" ("agentGraphExecutionId" text, "executionStatus" text)'
            )
            cursor.execute(
                'INSERT INTO platform."AgentGraphExecution" VALUES (%s, %s)',
                ("run", graph_status),
            )
            cursor.execute(
                'INSERT INTO platform."AgentNodeExecution" VALUES (%s, %s)',
                ("run", node_status),
            )
        connection.close()
        connection = None
    finally:
        if connection:
            connection.close()
        server.stop()
    before = smoke_test.snapshot(data.postgres)
    assert smoke_test.persisted_run(bundle.root, data.root, "graph/run") == expected
    assert smoke_test.changed_files(before, smoke_test.snapshot(data.postgres)) == []
    assert not (data.postgres / "postmaster.pid").exists()
