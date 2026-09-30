import logging
import multiprocessing
import signal
import threading
import time

from backend.util import process as process_module
from backend.util.process import STOP_TIMEOUT_SECONDS, AppProcess


def _ignore_sigterm_forever(ready):
    """A child that takes the SIGTERM and does nothing about it."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    ready.set()
    while True:
        time.sleep(0.05)


class _NoopProcess(AppProcess):
    def run(self):
        pass


class _FakeProcess:
    """Stands in for ``multiprocessing.Process`` so the joins can be inspected.

    ``alive_after_terminate`` says whether the child acts on the SIGTERM. One
    that does not must be escalated rather than waited on again.
    """

    def __init__(self, alive_after_terminate: bool):
        self.pid = 4242
        self.joins: list[float | None] = []
        self.terminated = False
        self.killed = False
        self._alive = True
        self._alive_after_terminate = alive_after_terminate

    def terminate(self):
        self.terminated = True
        self._alive = self._alive_after_terminate

    def kill(self):
        self.killed = True
        self._alive = False

    def is_alive(self):
        return self._alive

    def join(self, timeout=None):
        self.joins.append(timeout)


def test_stop_bounds_the_join_on_a_responsive_child():
    service = _NoopProcess()
    child = _FakeProcess(alive_after_terminate=False)
    service.process = child  # type: ignore[assignment]

    service.stop()

    assert child.terminated
    # The timeout is the whole point: an unbounded join here wedges whoever is
    # shutting the service down.
    assert child.joins == [STOP_TIMEOUT_SECONDS]
    assert not child.killed
    assert service.process is None


def test_stop_escalates_to_kill_when_the_child_ignores_the_signal(caplog):
    service = _NoopProcess()
    child = _FakeProcess(alive_after_terminate=True)
    service.process = child  # type: ignore[assignment]

    with caplog.at_level(logging.WARNING, logger=process_module.__name__):
        service.stop()

    assert child.killed
    assert child.joins == [STOP_TIMEOUT_SECONDS, STOP_TIMEOUT_SECONDS]
    assert "ignored SIGTERM" in caplog.text
    assert service.process is None


def test_stop_returns_even_when_a_real_child_ignores_sigterm(monkeypatch):
    """The same guarantee, against a real process and a real signal."""
    monkeypatch.setattr(process_module, "STOP_TIMEOUT_SECONDS", 2)

    service = _NoopProcess()
    ready = multiprocessing.Event()
    child = multiprocessing.Process(
        target=_ignore_sigterm_forever, args=(ready,), daemon=True
    )
    child.start()
    # Without this the child may simply have died on startup, and every
    # assertion below would pass for the wrong reason.
    assert ready.wait(timeout=30), "the child never came up ignoring SIGTERM"
    service.process = child

    returned = threading.Event()

    def _stop():
        service.stop()
        returned.set()

    # `stop` runs on a side thread so an unbounded join fails the assertion
    # below instead of hanging the whole session -- which is exactly how this
    # bug presented in the first place.
    threading.Thread(target=_stop, daemon=True).start()

    try:
        assert returned.wait(timeout=30), "stop() never returned: the join is unbounded"
        assert not child.is_alive()
        assert service.process is None
    finally:
        if child.is_alive():
            child.kill()
            child.join(timeout=10)


def test_stop_is_a_noop_without_a_process():
    service = _NoopProcess()
    assert service.process is None
    service.stop()
    assert service.process is None
