"""Tests for which questions Otto may ask, and how often."""

import pytest

from backend.copilot.model import ChatMessage
from backend.copilot.tools.ask_question import recommended_option
from backend.copilot.tools.models import ClarifyingQuestion
from backend.copilot.tools.question_rules import (
    LOOK_FIRST,
    QUESTION_BUDGET,
    questions_in_a_row,
    refusal,
)


def _q(text: str, reason: str | None = "goal_ambiguous") -> ClarifyingQuestion:
    return ClarifyingQuestion(
        question=text, keyword="k", reason=reason, default_guess="x"
    )


@pytest.mark.parametrize("mode", ["unsupervised", "auto", "ask_first"])
def test_a_blocking_question_may_be_asked_in_every_mode(mode):
    assert refusal([_q("Which of your 3 teams gets it?")], mode, 0) is None


def test_a_question_needs_a_reason():
    assert "give the reason" in (refusal([_q("Which team?", None)], "auto", 0) or "")


@pytest.mark.parametrize(
    "mode, allowed", [("unsupervised", False), ("auto", False), ("ask_first", True)]
)
def test_preferences_are_asked_only_in_ask_first(mode, allowed):
    result = refusal([_q("Formal or casual?", "preference")], mode, 0)
    assert (result is None) == allowed


@pytest.mark.parametrize(
    "text",
    [
        "Should I go ahead?",
        "Do you want me to also email them?",
        "OK to proceed?",
    ],
)
def test_permission_chatter_is_refused(text):
    assert "do not ask permission" in (refusal([_q(text)], "auto", 0) or "")


def test_should_i_inside_a_real_question_is_not_permission():
    assert refusal([_q("Which of the three bills should I pay?")], "auto", 0) is None


def test_asking_before_something_irreversible_is_allowed():
    question = _q("Should I delete the 3 old lists?", "irreversible_choice")
    assert refusal([question], "unsupervised", 0) is None


@pytest.mark.parametrize("mode, budget", sorted(QUESTION_BUDGET.items()))
def test_the_budget_caps_questions_in_a_row(mode, budget):
    question = [_q("Which team?")]
    assert refusal(question, mode, budget - 1) is None
    assert "in a row" in (refusal(question, mode, budget) or "")


def _user(text: str) -> ChatMessage:
    return ChatMessage(role="user", content=text)


def _assistant(*tools: str) -> ChatMessage:
    return ChatMessage(
        role="assistant",
        content="",
        tool_calls=[
            {"id": t, "type": "function", "function": {"name": t, "arguments": "{}"}}
            for t in tools
        ],
    )


def test_counts_only_the_unbroken_run_of_question_turns():
    messages = [
        _user("plan the launch"),
        _assistant("ask_question"),
        _user("nov 20"),
        _assistant("web_search"),
        _user("now the emails"),
        _assistant("web_search", "ask_question"),
        _user("casual"),
        _assistant("ask_question"),
        _user("sales team"),
    ]
    assert questions_in_a_row(messages) == 2


def test_a_fresh_chat_has_asked_nothing():
    assert questions_in_a_row([_user("hi")]) == 0
    assert questions_in_a_row([]) == 0


def test_a_question_before_any_look_is_sent_back_once():
    question = _q("Which bill?")
    assert refusal([question], "auto", 0, looked=False) == LOOK_FIRST
    assert refusal([question], "auto", 0, looked=True) is None


@pytest.mark.parametrize(
    "options, guess, expected",
    [
        (["Customers", "Everyone"], "customers", "Customers"),
        (["Customers", "Everyone"], "Send it to everyone on the list", "Everyone"),
        (["Customers", "Everyone"], "customers and everyone", None),
        (["Customers", "Everyone"], "the board", None),
        ([], "customers", None),
        (["Customers"], None, None),
    ],
)
def test_the_recommended_option_is_the_one_the_default_guess_names(
    options, guess, expected
):
    assert recommended_option(options, guess) == expected


@pytest.mark.parametrize("mode", ["unsupervised", "auto", "ask_first"])
def test_a_question_the_user_asked_for_is_asked_in_every_mode(mode):
    question = _q("Which tone should the note have?", "user_requested")
    assert refusal([question], mode, 0) is None


def test_a_refused_preference_says_how_to_ask_when_the_user_wanted_it():
    result = refusal([_q("Formal or casual?", "preference")], "auto", 0) or ""
    assert "user_requested" in result


@pytest.mark.parametrize("reason", ["goal_ambiguous", "user_only_fact", "conflict"])
def test_a_background_worker_with_a_workable_default_goes_on_with_it(reason):
    question = _q("Which three rivals should I cover?", reason)
    refused = refusal([question], "auto", 0, background=True) or ""

    assert "runs in the background" in refused and "default_guess" in refused
    assert "Which three rivals should I cover?" in refused
    # The same question in a chat the user is in may be asked.
    assert refusal([question], "auto", 0) is None


@pytest.mark.parametrize("reason", ["irreversible_choice", "user_requested"])
def test_a_background_worker_still_asks_before_what_cannot_be_undone(reason):
    question = _q("Refund all 14 orders or only the late one?", reason)
    assert refusal([question], "auto", 0, background=True) is None


def test_a_background_worker_with_no_default_at_all_may_ask():
    question = ClarifyingQuestion(
        question="Which bank account pays this?",
        keyword="k",
        reason="user_only_fact",
        default_guess="",
    )
    assert refusal([question], "auto", 0, background=True) is None
