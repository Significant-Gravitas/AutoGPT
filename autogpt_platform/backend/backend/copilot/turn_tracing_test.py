"""A turn's first token and first text are timed at the chunks that start them."""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from backend.copilot import turn_tracing
from backend.copilot.response_model import (
    StreamReasoningDelta,
    StreamTextDelta,
    StreamToolInputStart,
)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """The turn's monotonic clock, set by the test before each chunk."""
    now = [100.0]
    monkeypatch.setattr(
        turn_tracing,
        "time",
        SimpleNamespace(monotonic=lambda: now[0], time=time.time),
    )
    return now


@pytest.fixture
def timings(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[float]]:
    recorded: dict[str, list[float]] = {}

    def set_timing(name: str, milliseconds: float, *, span=None) -> None:
        recorded.setdefault(name, []).append(milliseconds)

    monkeypatch.setattr(turn_tracing.tracing, "set_timing", set_timing)
    return recorded


def test_first_token_and_text_are_timed_at_their_first_chunks(clock, timings):
    turn = turn_tracing.TurnTrace(MagicMock())

    for at, chunk in [
        (100.5, StreamReasoningDelta(id="r", delta="thinking")),
        (101.0, StreamTextDelta(id="t", delta="Here")),
        (101.5, StreamToolInputStart(toolCallId="c", toolName="run_block")),
        (105.0, StreamTextDelta(id="t2", delta="Done")),
    ]:
        clock[0] = at
        turn.observe(chunk)

    assert timings["copilot.turn.ttft_ms"] == [500.0]
    assert timings["copilot.turn.first_text_ms"] == [1000.0]
