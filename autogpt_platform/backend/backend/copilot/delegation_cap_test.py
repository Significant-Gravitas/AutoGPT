from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.copilot import delegation_cap
from backend.copilot.delegation_cap import (
    CAP_OPTIONS,
    CapAnswer,
    cap_question,
    parse_cap_answer,
)
from backend.copilot.model import PendingQuestion


def _parked(text: str | None = None, options: list[str] | None = None):
    return PendingQuestion(
        text=text or cap_question(2.0),
        asked_at=datetime.now(UTC),
        options=CAP_OPTIONS if options is None else options,
    )


def test_the_question_names_the_cap():
    assert cap_question(2) == (
        "This hand-off has reached its $2.00 cap. Raise it and continue?"
    )


@pytest.mark.parametrize(
    "message,expected",
    [
        ("Raise by $1", CapAnswer(raise_usd=1.0)),
        (" Raise by $5 ", CapAnswer(raise_usd=5.0)),
        ("stop", CapAnswer(raise_usd=None)),
        ("Raise it by 3 please", None),
    ],
)
def test_an_option_is_read_as_the_answer(message, expected):
    assert parse_cap_answer(_parked(), message) == expected


def test_only_the_cap_question_is_answered_this_way():
    other = _parked(text="Which release?", options=["Stop", "Raise by $1"])

    assert parse_cap_answer(other, "Stop") is None
    assert parse_cap_answer(None, "Stop") is None


@pytest.fixture
def seams(monkeypatch):
    db = MagicMock()
    db.raise_delegation_cap = AsyncMock(return_value=3.0)
    db.stop_delegation_at_cap = AsyncMock()
    cancel = AsyncMock()
    monkeypatch.setattr(delegation_cap, "delegation_db", lambda: db)
    monkeypatch.setattr(delegation_cap, "enqueue_cancel_task", cancel)
    return db, cancel


@pytest.mark.asyncio
async def test_a_raise_lifts_the_cap_and_resumes(seams):
    db, cancel = seams

    message = await delegation_cap.apply_cap_answer("t1", "u1", CapAnswer(raise_usd=1))

    db.raise_delegation_cap.assert_awaited_once_with("t1", "u1", 1)
    assert message is not None and "$3.00" in message and "Carry on" in message
    cancel.assert_not_awaited()


@pytest.mark.asyncio
async def test_stop_records_it_and_sends_nothing(seams):
    db, cancel = seams

    message = await delegation_cap.apply_cap_answer(
        "t1", "u1", CapAnswer(raise_usd=None)
    )

    assert message is None
    db.stop_delegation_at_cap.assert_awaited_once_with("t1", "u1")
    cancel.assert_awaited_once_with("t1")
