import contextlib
import logging
import threading

import pytest

from backend.util.llm import saturation

WORKERS = 3


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch):
    monkeypatch.setattr(saturation.settings.config, "num_graph_workers", WORKERS)
    monkeypatch.setattr(saturation, "_in_flight", {})
    monkeypatch.setattr(saturation, "_saturated", False)
    monkeypatch.setattr(saturation, "_executor_id", "exec-1")
    # Non-None keeps _ensure_ticker from spawning its daemon thread per test.
    monkeypatch.setattr(saturation, "_ticker", threading.current_thread())


@contextlib.contextmanager
def _warnings():
    """Attach to the module logger directly — caplog captures nothing for
    backend.* loggers under the app's logging config."""
    records: list[logging.LogRecord] = []

    class _Collect(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Collect(level=logging.DEBUG)
    saturation.logger.addHandler(handler)
    try:
        yield records
    finally:
        saturation.logger.removeHandler(handler)


@contextlib.contextmanager
def _busy(count: int, started: float, provider: str = "openai", first: int = 0):
    """`count` graph runs, each holding one LLM call started at `started`."""
    with contextlib.ExitStack() as stack:
        for i in range(first, first + count):
            stack.enter_context(
                saturation.track_llm_call(
                    graph_exec_id=f"run-{i}", provider=provider, now=started
                )
            )
        yield


def test_warns_once_while_saturation_persists():
    with _busy(WORKERS, started=0.0), _warnings() as records:
        assert saturation.evaluate_saturation(now=59.0) == 0, "not old enough yet"
        assert not records

        assert saturation.evaluate_saturation(now=61.0) == WORKERS
        assert len(records) == 1, "entering saturation must warn"

        saturation.evaluate_saturation(now=120.0)
        saturation.evaluate_saturation(now=600.0)
        assert len(records) == 1, "a persisting episode must not warn again"


def test_the_warning_carries_what_an_operator_needs():
    with _busy(2, started=0.0, provider="openai"), _busy(
        1, started=10.0, provider="anthropic", first=2
    ), _warnings() as records:
        saturation.evaluate_saturation(now=100.0)

    assert len(records) == 1
    # ERROR is the mechanism: LoggingIntegration()'s default event_level drops
    # anything lower, so a WARNING here would reach no alerting surface.
    assert records[0].levelno == logging.ERROR
    msg = records[0].getMessage()
    assert "exec-1" in msg, "an alert that can't name the pod can't be routed"
    assert "openai=2" in msg and "anthropic=1" in msg
    assert "longest_call_age=100s" in msg


def test_a_second_episode_warns_again():
    with _warnings() as records:
        with _busy(WORKERS, started=0.0):
            saturation.evaluate_saturation(now=61.0)
            assert len(records) == 1
        assert saturation.evaluate_saturation(now=62.0) == 0, "calls released"

        with _busy(WORKERS, started=100.0):
            saturation.evaluate_saturation(now=161.0)
    assert len(records) == 2, "a new episode is a new warning"


def test_one_free_worker_is_not_saturation():
    with _busy(WORKERS - 1, started=0.0), _warnings() as records:
        assert saturation.evaluate_saturation(now=600.0) == WORKERS - 1
        assert not records


def test_parallel_calls_in_one_run_occupy_one_worker():
    """A graph with two LLM nodes in flight still holds a single worker."""
    with contextlib.ExitStack() as stack, _warnings() as records:
        for _ in range(WORKERS * 2):
            stack.enter_context(
                saturation.track_llm_call(
                    graph_exec_id="run-0", provider="openai", now=0.0
                )
            )
        assert saturation.evaluate_saturation(now=600.0) == 1
        assert not records


def test_callers_without_a_graph_run_are_excluded():
    """Copilot, dream and briefing do not occupy graph workers."""
    with contextlib.ExitStack() as stack, _warnings() as records:
        for _ in range(WORKERS * 2):
            stack.enter_context(
                saturation.track_llm_call(
                    graph_exec_id=None, provider="openai", now=0.0
                )
            )
        assert saturation.evaluate_saturation(now=600.0) == 0
        assert not records


def test_the_episode_counter_tracks_the_warnings():
    before = saturation.saturation_episodes_counter._value.get()
    with _busy(WORKERS, started=0.0):
        saturation.evaluate_saturation(now=61.0)
        saturation.evaluate_saturation(now=62.0)
    assert saturation.saturation_episodes_counter._value.get() == before + 1


def test_the_first_tracked_call_starts_one_daemon_ticker(monkeypatch):
    """Every other test stubs `_ticker`, so this is the only place the thread the
    module rests on is really started."""
    monkeypatch.setattr(saturation, "_ticker", None)
    monkeypatch.setattr(saturation, "_TICK_SECONDS", 0.01)
    ticked = threading.Event()

    def _evaluate(now=None):
        # Park the daemon on its next sleep; the loop has no other exit.
        saturation._TICK_SECONDS = 3600
        ticked.set()
        return 0

    monkeypatch.setattr(saturation, "evaluate_saturation", _evaluate)

    with saturation.track_llm_call(graph_exec_id="run-0", provider="openai"):
        ticker = saturation._ticker
        with saturation.track_llm_call(graph_exec_id="run-1", provider="openai"):
            assert saturation._ticker is ticker, "one ticker per process, not per call"

    assert ticker is not None, "the first tracked call must start the ticker"
    assert ticker.daemon, "a non-daemon ticker would hold the process open at exit"
    assert ticked.wait(timeout=5), "the ticker never re-evaluated"


def test_the_ticker_keeps_ticking_after_a_failed_evaluation(monkeypatch):
    """A metrics thread that dies on one bad evaluation stops reporting for the
    life of the pod, silently."""
    monkeypatch.setattr(saturation, "_TICK_SECONDS", 0)
    calls: list[int] = []

    def _evaluate(now=None):
        calls.append(len(calls))
        if len(calls) == 1:
            raise RuntimeError("evaluation blew up")
        if len(calls) == 3:
            raise SystemExit  # BaseException: the only way out of `while True`
        return 0

    monkeypatch.setattr(saturation, "evaluate_saturation", _evaluate)
    with _warnings() as records, pytest.raises(SystemExit):
        saturation._tick_forever()

    assert len(calls) == 3, "one failed evaluation must not stop the timer"
    assert [r.levelno for r in records] == [logging.ERROR], "the failure must be logged"
