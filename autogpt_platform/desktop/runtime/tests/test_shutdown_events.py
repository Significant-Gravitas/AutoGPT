import io
import json
import math
from types import SimpleNamespace

import pytest

from autogpt_desktop import events, runs, supervisor
from autogpt_desktop.layout import Bundle, DataDir

ACTIVE = runs.Activity(
    ("private-run",), ("private-session",), (("private-session", "private-turn"),)
)
STORED = "stored-running"


class Clock:
    def __init__(self):
        self.now = 100.0
        self.on_read = None

    def monotonic(self):
        now = self.now
        if self.on_read:
            self.on_read()
        return now


class Hurry:
    def __init__(self, clock):
        self.clock = clock
        self.requested = False
        self.after_wait = None
        self.waits = []
        self.checks = 0

    def set(self):
        self.requested = True

    def is_set(self):
        self.checks += 1
        return self.requested

    def wait(self, timeout):
        self.waits.append(timeout)
        assert len(self.waits) <= 50, "cancellation exceeded its bounded wait"
        self.clock.now += timeout
        if self.after_wait:
            self.after_wait()
        return self.requested


class Cache:
    def __init__(self, clock, order):
        self.clock = clock
        self.order = order
        self.activity = ACTIVE
        self.ends_at = math.inf
        self.asked = 0

    def active(self):
        self.asked += 1
        return self.activity if self.clock.now < self.ends_at else runs.Activity()

    def close(self):
        self.order.append("cache closed")


@pytest.fixture
def shutdown(tmp_path, monkeypatch):
    data = DataDir(tmp_path / "data")
    data.prepare()
    stack = supervisor.Stack(Bundle(tmp_path / "bundle"), data)
    clock = Clock()
    order = []
    cache = Cache(clock, order)
    hurry = Hurry(clock)
    wire = io.StringIO()
    cancelled = []
    stored = [STORED]
    record_stopped = runs.record_stopped

    def record(path, activity):
        order.append("recorded")
        record_stopped(path, activity)
        clock.now += 7.25

    def cancel(activity):
        assert runs.read_stopped(stack.stopped_runs) == activity
        order.append("cancelled")
        cancelled.append(activity)
        clock.now += 19.75

    def running(connect, among=None):
        assert connect is stack.connect
        return tuple(stored) if among is None else ()

    monkeypatch.setattr(supervisor, "time", clock)
    monkeypatch.setattr(events, "sys", SimpleNamespace(stdout=wire))
    monkeypatch.setattr(runs, "record_stopped", record)
    monkeypatch.setattr(runs, "running", running)
    stack.cache = cache
    stack.hurry = hurry
    stack.queue = SimpleNamespace(cancel=cancel)
    stack.connect = lambda: pytest.fail("must not open a database connection")
    return SimpleNamespace(
        stack=stack,
        clock=clock,
        cache=cache,
        hurry=hurry,
        wire=wire,
        order=order,
        cancelled=cancelled,
        stored=stored,
    )


def wire_events(shutdown):
    return [json.loads(line) for line in shutdown.wire.getvalue().splitlines()]


def assert_cancellation(shutdown, outcome, seconds):
    stopping, cancellation = wire_events(shutdown)
    assert stopping["event"] == "progress"
    assert stopping["step"] == "stopping"
    assert stopping["grace_seconds"] == 120
    assert cancellation == {
        "event": "shutdown",
        "phase": "cancellation",
        "seconds": seconds,
        "outcome": outcome,
    }
    assert math.isfinite(cancellation["seconds"])
    assert cancellation["seconds"] >= 0
    assert supervisor.CANCEL_WAIT_SECONDS == 20
    assert supervisor.STOPPING_RUNS_GRACE_SECONDS == 120
    assert supervisor.SHELL_STOP_GRACE_SECONDS == 60
    assert shutdown.hurry.waits == [0.5] * len(shutdown.hurry.waits)
    chosen = shutdown.cache.activity.merged(runs.Activity(tuple(shutdown.stored)))
    assert shutdown.stack.stopping == chosen
    assert runs.read_stopped(shutdown.stack.stopped_runs) == chosen
    assert shutdown.cancelled == [chosen]
    assert shutdown.order[:2] == ["recorded", "cancelled"]
    return cancellation


def test_cancelled_wait_excludes_intent_and_queue_delay(shutdown):
    shutdown.cache.ends_at = 128.0

    assert shutdown.stack.stop_runs() is True

    assert_cancellation(shutdown, "cancelled", 1.0)
    assert len(shutdown.hurry.waits) == 2


def test_frozen_wait_gets_full_deadline_after_intent_and_queue_delay(shutdown):
    assert shutdown.stack.stop_runs() is True

    assert_cancellation(shutdown, "deadline", 20.0)
    assert shutdown.clock.now == 147.0
    assert len(shutdown.hurry.waits) == 40
    assert shutdown.cache.asked == 41


def test_second_stop_request_hurries_the_actual_wait(shutdown):
    shutdown.stack.request_stop(stop_the_runs=True)
    shutdown.hurry.after_wait = shutdown.stack.request_stop

    assert shutdown.stack.stop_runs() is True

    assert_cancellation(shutdown, "hurried", 0.5)
    assert shutdown.stack.stop_the_runs
    assert len(shutdown.hurry.waits) == 1
    assert shutdown.cache.asked == 2


def test_already_hurried_stop_still_records_and_cancels(shutdown):
    shutdown.hurry.set()

    assert shutdown.stack.stop_runs() is True

    assert_cancellation(shutdown, "hurried", 0.0)
    assert shutdown.hurry.waits == []
    assert shutdown.cache.asked == 1


def test_stored_running_with_no_active_work_is_inactive_even_when_hurried(shutdown):
    shutdown.cache.activity = runs.Activity()
    shutdown.hurry.set()

    assert shutdown.stack.stop_runs() is True

    assert_cancellation(shutdown, "inactive", 0.0)
    assert shutdown.hurry.waits == []
    assert shutdown.hurry.checks == 0


def test_no_work_emits_nothing_and_returns_false(shutdown):
    shutdown.cache.activity = runs.Activity()
    shutdown.stored.clear()

    assert shutdown.stack.stop_runs() is False

    assert wire_events(shutdown) == []
    assert shutdown.order == []
    assert shutdown.cancelled == []
    assert shutdown.hurry.waits == []
    assert not shutdown.stack.stopped_runs.exists()
    assert shutdown.stack.stopping == runs.Activity()
    assert shutdown.clock.now == 100.0


def test_cancellation_seconds_keep_subcentisecond_precision(shutdown, monkeypatch):
    def stopped(active):
        assert active == ACTIVE
        shutdown.clock.now += 0.00390625
        return True

    monkeypatch.setattr(shutdown.stack, "has_stopped", stopped)

    assert shutdown.stack.stop_runs() is True

    assert_cancellation(shutdown, "cancelled", 0.00390625)


def test_success_wins_if_hurry_and_deadline_arrive_during_has_stopped(
    shutdown, monkeypatch
):
    def stopped(active):
        shutdown.clock.now += 21.0
        shutdown.hurry.set()
        return True

    monkeypatch.setattr(shutdown.stack, "has_stopped", stopped)

    assert shutdown.stack.stop_runs() is True

    assert_cancellation(shutdown, "cancelled", 21.0)
    assert shutdown.hurry.waits == []


def test_deadline_outcome_is_not_reclassified_by_a_late_hurry(shutdown):
    def late_hurry():
        if shutdown.clock.now >= 147.0:
            shutdown.hurry.set()

    shutdown.clock.on_read = late_hurry

    assert shutdown.stack.stop_runs() is True

    assert_cancellation(shutdown, "deadline", 20.0)
    assert shutdown.hurry.requested


def test_hurry_is_checked_before_an_already_expired_deadline(shutdown):
    def hurry_at_deadline():
        if shutdown.clock.now >= 147.0:
            shutdown.hurry.set()

    shutdown.hurry.after_wait = hurry_at_deadline

    assert shutdown.stack.stop_runs() is True

    assert_cancellation(shutdown, "hurried", 20.0)


def test_has_stopped_exception_is_preserved_without_a_false_completion(
    shutdown, monkeypatch
):
    failure = RuntimeError("stop check failed")

    def fail(active):
        raise failure

    monkeypatch.setattr(shutdown.stack, "has_stopped", fail)

    with pytest.raises(RuntimeError) as raised:
        shutdown.stack.stop_runs()

    assert raised.value is failure
    assert [event["event"] for event in wire_events(shutdown)] == ["progress"]
    assert shutdown.hurry.waits == []
    assert shutdown.cancelled == [shutdown.stack.stopping]
    assert runs.read_stopped(shutdown.stack.stopped_runs) == shutdown.stack.stopping


def fake_teardown(shutdown, monkeypatch, before_proxy):
    stack = shutdown.stack
    stack.ready_at = 1.0
    stack.request_stop(stop_the_runs=True)
    services, servers = [["workers"], ["frontend"]], [["database"]]

    def proxy_stop():
        before_proxy()
        shutdown.order.append("proxy stopped")
        shutdown.clock.now += 11.0

    def stop_together(tier):
        shutdown.order.append(tuple(tier))
        shutdown.clock.now += 30.0

    def end_stopped(connect, path, cache, queue, chosen):
        assert connect is stack.connect
        assert path == stack.stopped_runs
        assert cache is stack.cache
        assert queue is stack.queue
        assert chosen == stack.stopping
        shutdown.order.append("ended")

    stack.proxy = SimpleNamespace(stop=proxy_stop)
    monkeypatch.setattr(stack, "tiers_to_stop", lambda: (services, servers))
    monkeypatch.setattr(supervisor, "stop_together", stop_together)
    monkeypatch.setattr(runs, "end_stopped", end_stopped)


def assert_teardown(shutdown):
    assert shutdown.order == [
        "recorded",
        "cancelled",
        "proxy stopped",
        ("frontend",),
        ("workers",),
        "ended",
        "cache closed",
        ("database",),
    ]


def test_frozen_wait_event_precedes_proxy_and_service_teardown(shutdown, monkeypatch):
    fake_teardown(
        shutdown,
        monkeypatch,
        lambda: assert_cancellation(shutdown, "deadline", 20.0),
    )

    shutdown.stack.stop()

    assert_teardown(shutdown)
    assert shutdown.clock.now == 248.0
    assert_cancellation(shutdown, "deadline", 20.0)


@pytest.mark.parametrize("operation", ["write", "flush"])
@pytest.mark.parametrize("error", [BrokenPipeError, OSError, ValueError])
def test_broken_output_during_diagnostic_does_not_interrupt_shutdown(
    shutdown, monkeypatch, operation, error
):
    attempts = []

    class BrokenDiagnostic(io.StringIO):
        def write(self, line):
            if operation == "write" and '"event":"shutdown"' in line:
                attempts.append(operation)
                raise error("shell closed")
            return super().write(line)

        def flush(self):
            if operation == "flush" and '"event":"shutdown"' in self.getvalue():
                attempts.append(operation)
                raise error("shell closed")
            return super().flush()

    wire = BrokenDiagnostic()
    monkeypatch.setattr(events, "sys", SimpleNamespace(stdout=wire))
    fake_teardown(shutdown, monkeypatch, lambda: None)

    shutdown.stack.stop()

    assert attempts == [operation]
    assert_teardown(shutdown)
    assert shutdown.clock.now == 248.0
    assert runs.read_stopped(shutdown.stack.stopped_runs) == shutdown.stack.stopping
