"""Tests for the task eval: the simulated world, the checks and the turn loop."""

import json
from typing import Any

import pytest

from .judge import run_checks, score_case
from .llm import Call, Reply, cli_result, parse_reply, render_chat_prompt
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


def _message(text: str = "", calls: list[tuple[str, dict]] | None = None) -> Reply:
    return Reply(
        content=text,
        tool_calls=[
            Call(id=f"t{i}", name=name, arguments=json.dumps(args))
            for i, (name, args) in enumerate(calls or [])
        ],
    )


class FakeBackend:
    """Answers Otto's model calls and the simulated user's from scripts."""

    def __init__(self, replies: list[Reply], answers: list[str] | None = None):
        self.replies, self.answers = list(replies), list(answers or [])

    async def chat(self, model, messages, tools) -> Reply:
        return self.replies.pop(0)

    async def text(self, model, system, user, json_out) -> str:
        return self.answers.pop(0)


def _models(*replies: Reply, answers: list[str] | None = None) -> Models:
    return Models(FakeBackend(list(replies), answers), otto="otto", user="user")


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
        _message("Sent to sales."),
        answers=["sales"],
    )

    run = await run_case(_case(), WORLD, models)

    first, answered = run.turns
    assert first.ended_by == "question"
    assert [q.question for q in first.questions] == ["Which team?"]
    assert first.tool_calls[1].refused is not None
    assert answered.user == "sales"
    assert answered.reply == "Sent to sales."


@pytest.mark.asyncio
async def test_after_a_started_task_only_tasks_and_a_closing_line_follow():
    models = _models(
        _message(
            "",
            [
                ("start_task", {"title": "Plan funnel", "brief": "b"}),
                ("start_task", {"title": "Fix bug", "brief": "b"}),
            ],
        ),
        _message("", [("web_search", {"query": "funnels"})]),
        _message("Both are running; results land here."),
    )

    run = await run_case(_case(turns=[{"user": "plan it and fix it"}]), WORLD, models)

    (turn,) = run.turns
    assert turn.ended_by == "task_started"
    assert [c.name for c in turn.tool_calls] == [
        "start_task",
        "start_task",
        "web_search",
    ]
    assert turn.tool_calls[2].refused is not None
    assert turn.reply == "Both are running; results land here."


def test_delegating_counts_a_task_the_expert_owns():
    case = _case(expected={"checks": {"delegates": True}})

    def delegated(*calls: ToolCall) -> bool:
        run = Run(case_id="c1", turns=[TurnRecord(user="hi", tool_calls=list(calls))])
        return all(c.passed for c in run_checks(case, run))

    assert delegated(ToolCall(name="delegate_to_expert"))
    assert delegated(ToolCall(name="start_task", arguments={"expert_id": "exp_1"}))
    assert not delegated(ToolCall(name="start_task", arguments={"title": "x"}))


def test_the_cursor_answer_is_read_from_its_json_protocol():
    reply = parse_reply(
        'Sure.\n```json\n{"text": "On it.", "tool_calls": '
        '[{"name": "start_task", "arguments": {"title": "Plan"}}]}\n```'
    )

    assert reply.content == "On it."
    assert [(c.name, json.loads(c.arguments)) for c in reply.tool_calls] == [
        ("start_task", {"title": "Plan"})
    ]
    assert parse_reply("just words").content == "just words"


def test_the_prompt_carries_tools_turns_and_the_protocol():
    prompt = render_chat_prompt(
        [
            {"role": "system", "content": "You are Otto."},
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "t0", "function": {"name": "web_search", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "t0", "content": "results"},
        ],
        [
            {
                "function": {
                    "name": "web_search",
                    "description": "Search",
                    "parameters": {},
                }
            }
        ],
    )

    assert prompt.startswith("You are Otto.")
    assert '"name":"web_search"' in prompt
    assert "[user]\nhi" in prompt and "[tool result t0]\nresults" in prompt
    assert '"tool_calls": []' in prompt


def test_the_cli_result_line_is_found_and_errors_are_none():
    out = 'noise\n{"type":"result","is_error":false,"result":"{\\"text\\":\\"hi\\"}"}\n'
    assert cli_result(out) == '{"text":"hi"}'
    assert cli_result('{"type":"result","is_error":true,"result":"x"}') is None
    assert cli_result("not json") is None


def test_claude_style_invoke_blocks_count_as_calls():
    reply = parse_reply(
        'Checking your inbox.\n<function_calls>\n<invoke name="run_capability">\n'
        '<parameter name="id">service:gmail/search</parameter>\n'
        '<parameter name="input">{"q": "invoice"}</parameter>\n</invoke>\n</function_calls>'
    )

    assert reply.content == "Checking your inbox."
    assert [(c.name, json.loads(c.arguments)) for c in reply.tool_calls] == [
        ("run_capability", {"id": "service:gmail/search", "input": {"q": "invoice"}})
    ]
