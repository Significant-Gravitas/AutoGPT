"""Tests for the task eval: the simulated world, the checks and the turn loop."""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall

from .judge import run_checks, score_case
from .models import Case, Review, Run, ToolCall, TurnRecord, World
from .runner import Models, run_case
from .simulator import Simulator

WORLD = World.model_validate(
    {
        "id": "w",
        "user": {"name": "Maya", "writing_style": "short"},
        "connected_accounts": [{"provider": "gmail", "account": "maya@x.co"}],
        "not_connected": ["hubspot"],
        "data": {
            "gmail": {
                "messages": [
                    {
                        "id": "m1",
                        "subject": "Packaging delay",
                        "body": "two weeks late",
                    },
                    {"id": "m2", "subject": "Newsletter", "body": "sale"},
                ]
            }
        },
        "web": {"search": [{"keywords": ["delhi"], "results": [{"title": "Delhi"}]}]},
    }
)


def _case(**overrides: Any) -> Case:
    return Case.model_validate(
        {
            "id": "c1",
            "category": "test",
            "world": "w",
            "turns": [{"user": "hi"}],
            "answer_sheet": [{"topic": "team", "answer": "sales"}],
        }
        | overrides
    )


def test_a_missing_account_answers_with_the_sign_in_card():
    call, result = Simulator(WORLD).call(
        "run_capability", {"id": "service:hubspot/search", "input": {}}
    )
    assert call.service == "hubspot"
    assert json.loads(result)["type"] == "setup_requirements"


def test_reads_search_the_world_and_writes_are_only_recorded():
    sim = Simulator(WORLD)
    _, found = sim.call(
        "run_capability", {"id": "service:gmail/search", "input": {"q": "packaging"}}
    )
    assert [i["id"] for i in json.loads(found)["items"]] == ["m1"]

    _, sent = sim.call(
        "run_capability", {"id": "service:gmail/send", "input": {"to": "a@b.c"}}
    )
    assert json.loads(sent)["status"] == "ok"
    assert sim.writes == [
        {"service": "gmail", "id": "service:gmail/send", "input": {"to": "a@b.c"}}
    ]


def test_platform_tools_reached_by_id_are_recorded_by_name():
    call, result = Simulator(WORLD).call(
        "run_capability", {"id": "tool:list_tasks", "input": {}}
    )
    assert call.name == "list_tasks"
    assert json.loads(result) == {"tasks": []}


def test_checks_compare_the_run_with_the_case():
    case = _case(
        expected={
            "checks": {
                "tools_called": ["start_task"],
                "asks_question": False,
                "max_tool_calls": 1,
            }
        }
    )
    run = Run(
        case_id="c1",
        turns=[TurnRecord(user="hi", tool_calls=[ToolCall(name="web_search")])],
    )

    results = {c.name: c.passed for c in run_checks(case, run)}

    assert results == {
        "max_tool_calls": True,
        "called:start_task": False,
        "asks_question": True,
    }


def test_a_failed_check_caps_a_pass_at_partial():
    case = _case(expected={"checks": {"tools_called": ["start_task"]}})
    run = Run(case_id="c1", turns=[TurnRecord(user="hi")])
    checks = run_checks(case, run)

    result = score_case(case, run, checks, Review(verdict="pass", score=10))

    assert result.verdict == "partial"
    assert result.score == 50.0


def _message(text: str = "", calls: list[tuple[str, dict]] | None = None):
    tool_calls = [
        ChatCompletionMessageToolCall(
            id=f"t{i}",
            type="function",
            function={"name": name, "arguments": json.dumps(args)},
        )
        for i, (name, args) in enumerate(calls or [])
    ]
    return ChatCompletionMessage(
        role="assistant", content=text, tool_calls=tool_calls or None
    )


def _models(*messages: Any) -> Models:
    """A client that answers each model call with the next message in turn
    (Otto's and the simulated user's calls share it)."""
    replies = [MagicMock(choices=[MagicMock(message=m)], usage=None) for m in messages]
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=replies)
    return Models(client, otto="otto", user="user")


QUESTION = {
    "questions": [
        {
            "question": "Which team?",
            "reason": "goal_ambiguous",
            "default_guess": "sales",
        }
    ]
}


@pytest.mark.asyncio
async def test_a_question_ends_the_turn_and_refuses_the_calls_after_it():
    models = _models(
        _message(calls=[("ask_question", QUESTION), ("web_search", {"query": "x"})]),
        MagicMock(content="sales"),
        _message("Sent to sales."),
    )

    run = await run_case(_case(), WORLD, models)

    first, answered = run.turns
    assert first.ended_by == "question"
    assert [q.question for q in first.questions] == ["Which team?"]
    assert first.tool_calls[1].refused is not None
    assert answered.user == "sales"
    assert answered.reply == "Sent to sales."


@pytest.mark.asyncio
async def test_a_started_task_ends_the_turn_after_its_step():
    models = _models(
        _message(
            "On it.",
            [
                ("start_task", {"title": "Plan funnel", "brief": "b"}),
                ("start_task", {"title": "Fix bug", "brief": "b"}),
            ],
        )
    )

    run = await run_case(_case(turns=[{"user": "plan it and fix it"}]), WORLD, models)

    (turn,) = run.turns
    assert turn.ended_by == "task_started"
    assert [c.name for c in turn.tool_calls] == ["start_task", "start_task"]
    assert turn.rounds == 1
