"""The interview's scope guard on the server: the redirect count, and the
classifier seam that feeds it."""

import asyncio
from collections.abc import Sequence
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot import onboarding_interview_scope as scope
from backend.copilot.model import ONBOARDING_INTERVIEW_KIND, ChatMessage, ChatSession
from backend.copilot.onboarding_interview import (
    ONBOARDING_INTERVIEW_KICKOFF_MESSAGE,
    onboarding_interview_kickoff_metadata,
)
from backend.copilot.onboarding_interview_scope import (
    ScopedTurn,
    ScopeLabel,
    judge_interview_scope,
    recent_turns,
    scope_verdict,
)


def _interview(*rows: ChatMessage) -> ChatSession:
    session = ChatSession.new("user-1", dry_run=False)
    session.metadata.kind = ONBOARDING_INTERVIEW_KIND
    session.messages = list(rows)
    return session


# ---------- the rule ----------


def test_three_requests_in_a_row_close_the_interview() -> None:
    """The ask, then a push past each of two redirects: the prompt's line."""
    assert scope_verdict(["request", "request"]) is None
    assert scope_verdict(["request", "request", "request"]) == "scope"


def test_a_turn_about_their_work_clears_the_count() -> None:
    labels: list[ScopeLabel] = ["request", "request", "work", "request", "request"]

    assert scope_verdict(labels) is None


def test_a_turn_about_nothing_in_particular_leaves_the_count() -> None:
    labels: list[ScopeLabel] = ["request", "other", "request", "other", "request"]

    assert scope_verdict(labels) == "scope"


def test_a_harmful_ask_pushed_past_one_refusal_closes_it() -> None:
    assert scope_verdict(["harmful"]) is None
    assert scope_verdict(["harmful", "harmful"]) == "scope"


def test_a_harmful_ask_counts_toward_the_requests_too() -> None:
    assert scope_verdict(["request", "harmful", "request"]) == "scope"


def test_cooperative_answers_never_close_it() -> None:
    assert scope_verdict(["work"] * 16) is None
    assert scope_verdict([]) is None


# ---------- the turns a classifier sees ----------


def test_each_turn_is_shown_with_the_line_of_otto_s_it_answered() -> None:
    session = _interview(
        ChatMessage(
            role="user",
            content=ONBOARDING_INTERVIEW_KICKOFF_MESSAGE,
            metadata=onboarding_interview_kickoff_metadata(),
        ),
        ChatMessage(role="assistant", content="Hi, I'm Otto, an AI. What do you do?"),
        ChatMessage(role="user", content="Fix my Flask app."),
        ChatMessage(role="assistant", content="", tool_calls=[{"id": "t"}]),
        ChatMessage(role="tool", content="{}", tool_call_id="t"),
        ChatMessage(role="assistant", content="That's one for your team. Your role?"),
        ChatMessage(role="user", content="Just fix it."),
    )

    assert recent_turns(session) == [
        ScopedTurn(
            otto="Hi, I'm Otto, an AI. What do you do?", user="Fix my Flask app."
        ),
        ScopedTurn(otto="That's one for your team. Your role?", user="Just fix it."),
    ]


def test_only_the_latest_turns_are_shown() -> None:
    rows: list[ChatMessage] = []
    for n in range(6):
        rows.append(ChatMessage(role="assistant", content=f"Question {n}?"))
        rows.append(ChatMessage(role="user", content=f"Answer {n}."))

    turns = recent_turns(_interview(*rows))

    assert [turn.user for turn in turns] == ["Answer 3.", "Answer 4.", "Answer 5."]


# ---------- the classifier seam ----------


def test_the_classifier_is_off_by_default() -> None:
    assert scope.interview_scope_classifier() is None


@pytest.mark.asyncio
async def test_with_the_classifier_off_no_turn_is_judged() -> None:
    session = MagicMock()

    assert await judge_interview_scope(session) is None
    # Not even read: off costs nothing.
    assert session.mock_calls == []


def _classifier(labels: Sequence[ScopeLabel]) -> AsyncMock:
    return AsyncMock(return_value=list(labels))


@pytest.mark.asyncio
async def test_the_classifier_s_labels_go_through_the_rule() -> None:
    session = _interview(ChatMessage(role="user", content="Write my essay."))
    classifier = _classifier(["request", "request", "request"])
    with patch.object(scope, "interview_scope_classifier", return_value=classifier):
        assert await judge_interview_scope(session) == "scope"

    classifier.assert_awaited_once_with([ScopedTurn(otto="", user="Write my essay.")])


@pytest.mark.asyncio
async def test_a_cooperative_run_is_not_closed() -> None:
    session = _interview(ChatMessage(role="user", content="I run a bakery."))
    classifier = _classifier(["request", "work"])
    with patch.object(scope, "interview_scope_classifier", return_value=classifier):
        assert await judge_interview_scope(session) is None


@pytest.mark.asyncio
async def test_a_classifier_that_fails_never_ends_an_interview() -> None:
    session = _interview(ChatMessage(role="user", content="Write my essay."))
    broken = AsyncMock(side_effect=RuntimeError("model down"))
    with patch.object(scope, "interview_scope_classifier", return_value=broken):
        assert await judge_interview_scope(session) is None


@pytest.mark.asyncio
async def test_a_classifier_that_runs_out_of_time_never_ends_an_interview() -> None:
    async def stalled(_turns: Sequence[ScopedTurn]) -> list[ScopeLabel]:
        await asyncio.sleep(10)
        return ["request", "request", "request"]

    session = _interview(ChatMessage(role="user", content="Write my essay."))
    with (
        patch.object(scope, "interview_scope_classifier", return_value=stalled),
        patch.object(scope, "CLASSIFIER_TIMEOUT_SECONDS", 0.01),
    ):
        assert await judge_interview_scope(session) is None
