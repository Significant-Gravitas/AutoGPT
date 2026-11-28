"""What the spend-alert tests (#16173) share: a fake clock for the llm module,
which only the test moves, and the records the module logs."""

import logging
import time
from collections.abc import Iterator
from typing import Any

import pytest

import backend.blocks.llm as llm

PROMPT = [{"role": "user", "content": "Hello"}]
_BUDGET = 600


class Clock:
    def __init__(self) -> None:
        # A real base keeps the saturation tracker's ages sane.
        self.start = float(int(time.monotonic()))
        self.now = self.start

    def at(self, seconds: float) -> None:
        self.now = self.start + seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    clock = Clock()
    monkeypatch.setattr(llm, "_now", lambda: clock.now)
    monkeypatch.setattr(llm, "LLM_REQUEST_TIMEOUT_SECONDS", _BUDGET)
    return clock


@pytest.fixture
def logs() -> Iterator[list[logging.LogRecord]]:
    """What the llm module logs during the test. The handler sits on the
    module's own logger: caplog's root handler sees nothing under the app's
    logging config."""
    records: list[logging.LogRecord] = []

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    target = logging.getLogger(llm.__name__)
    handler = _Collect(level=logging.DEBUG)
    previous = target.level
    target.addHandler(handler)
    target.setLevel(logging.DEBUG)
    try:
        yield records
    finally:
        target.removeHandler(handler)
        target.setLevel(previous)


def alert_fields(records: list[logging.LogRecord]) -> list[dict[str, Any]]:
    # Containment: the app's log config can wrap the message in colour codes.
    return [
        vars(r)["json_fields"]
        for r in records
        if r.levelno == logging.ERROR and llm.SPEND_BURNED_ALERT in r.getMessage()
    ]


def timeout_warning(records: list[logging.LogRecord]) -> str:
    warnings = [
        r.getMessage()
        for r in records
        if r.levelno == logging.WARNING and "LLM call timed out" in r.getMessage()
    ]
    assert len(warnings) == 1, warnings
    return warnings[0]
