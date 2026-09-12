"""The tool loop without a provider: stubbed tool results feed the next
round, ask_question ends the turn, usage and cost accumulate."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai.types.chat import ChatCompletion

from backend.copilot.config import ChatConfig
from backend.copilot.tools.models import ResponseType

from .assembly import attach_workflows, load_fixtures, roster_experts
from .generation import (
    MAX_TOOL_ROUNDS,
    add_usage,
    expert_tools,
    generate_turn,
    question_text,
    stub_tool_result,
    usage_of,
)
from .models import Usage

ROSTER = roster_experts()


def _completion(
    text: str | None,
    tool_calls: list[tuple[str, str, dict]] | None = None,
    *,
    cost: float | None = 0.01,
    cached: int = 0,
) -> ChatCompletion:
    message: dict = {"role": "assistant", "content": text}
    if tool_calls:
        message["tool_calls"] = [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
            for call_id, name, args in tool_calls
        ]
    usage: dict = {
        "prompt_tokens": 100 + cached,
        "completion_tokens": 20,
        "total_tokens": 120 + cached,
        "prompt_tokens_details": {"cached_tokens": cached},
    }
    if cost is not None:
        usage["cost"] = cost
    return ChatCompletion.model_validate(
        {
            "id": "c",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                    "message": message,
                }
            ],
            "usage": usage,
        }
    )


def _client(*completions: ChatCompletion) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=list(completions))
    return client


@pytest.mark.asyncio
async def test_tool_calls_get_a_stub_result_and_the_loop_continues():
    client = _client(
        _completion("Checking.", [("t1", "memory_search", {"query": "x"})]),
        _completion("Here is the answer."),
    )
    turn = await generate_turn(
        client,
        ChatConfig(),
        model="anthropic/claude-x",
        expert=roster_experts(["Max"])[0],
        roster=ROSTER,
        user_message="hi",
    )
    assert turn.text == "Checking.\n\nHere is the answer."
    assert turn.tool_calls == ["memory_search"]
    second_call = client.chat.completions.create.await_args_list[1].kwargs
    assert second_call["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "t1",
        "content": stub_tool_result("memory_search", "{}", None, ROSTER),
    }
    assert turn.rounds == 2
    assert turn.finish_reasons == ["tool_calls", "stop"]
    assert not turn.hit_round_cap
    assert second_call["messages"][-2]["role"] == "assistant"
    assert turn.usage.cost_usd == pytest.approx(0.02)
    assert turn.usage.input_tokens == 200


@pytest.mark.asyncio
async def test_ask_question_ends_the_turn_with_the_question_as_text():
    client = _client(
        _completion(
            "One thing first.",
            [
                (
                    "t1",
                    "ask_question",
                    {"questions": [{"question": "Which list?", "options": ["A", "B"]}]},
                )
            ],
        ),
    )
    turn = await generate_turn(
        client, ChatConfig(), model="m", expert=None, roster=ROSTER, user_message="hi"
    )
    assert turn.text == "One thing first.\n\nWhich list? (A / B)"
    assert turn.tool_calls == ["ask_question"]
    assert client.chat.completions.create.await_count == 1


@pytest.mark.asyncio
async def test_round_cap_stops_a_turn_that_never_answers():
    looping = [_completion(None, [("t", "memory_search", {})])] * (MAX_TOOL_ROUNDS + 2)
    client = _client(*looping)
    turn = await generate_turn(
        client, ChatConfig(), model="m", expert=None, roster=ROSTER, user_message="hi"
    )
    assert turn.text == ""
    assert turn.hit_round_cap
    assert turn.rounds == MAX_TOOL_ROUNDS
    assert client.chat.completions.create.await_count == MAX_TOOL_ROUNDS


def _stub(name: str, args: dict | None = None) -> dict:
    (fixture,) = load_fixtures(["Max"])
    expert = attach_workflows(roster_experts(["Max"])[0], fixture)
    return json.loads(stub_tool_result(name, json.dumps(args or {}), expert, ROSTER))


def test_library_search_stub_lists_the_experts_installed_workflows():
    (fixture,) = load_fixtures(["Max"])
    found = _stub("find_library_agent")
    assert [a["name"] for a in found["agents"]] == [w.name for w in fixture.workflows]
    assert found["count"] == 3
    assert (
        json.loads(stub_tool_result("find_library_agent", "{}", None, ROSTER))["type"]
        == ResponseType.NO_RESULTS
    )


def test_run_agent_stub_queues_the_run_the_call_named():
    (fixture,) = load_fixtures(["Max"])
    expert = attach_workflows(roster_experts(["Max"])[0], fixture)
    wanted = expert.workflows[1]
    started = _stub("run_agent", {"library_agent_id": wanted.library_agent_id})
    assert started["type"] == ResponseType.EXECUTION_STARTED
    assert started["graph_name"] == wanted.name
    assert started["status"] == "QUEUED"
    assert _stub("run_agent")["graph_name"] == expert.workflows[0].name


@pytest.mark.parametrize(
    "name",
    [
        "run_agent",
        "view_agent_output",
        "memory_search",
        "delegate_to_expert",
        "handoff_to_expert",
        "list_team",
        "list_schedules",
        "list_presets",
        "list_agent_triggers",
        "list_workspace_files",
        "web_search",
        "read_workspace_file",
    ],
)
def test_every_stub_answers_in_a_production_response_shape(name: str):
    """A hand-rolled ``{"success": true, "message": "No results."}`` matched no
    tool's real output and kept the model retrying until the round cap."""
    stub = _stub(name)
    assert stub["type"] in {t.value for t in ResponseType}
    assert stub["message"]
    assert "success" not in stub


def test_list_team_stub_names_the_roster():
    assert [e["name"] for e in _stub("list_team")["experts"]] == [
        e.name for e in ROSTER
    ]


def test_usage_of_prices_openrouter_cost_and_splits_cached_tokens():
    usage = usage_of("anthropic/claude-x", _completion("t", cost=0.5, cached=60))
    assert (usage.input_tokens, usage.cache_read_tokens) == (100, 60)
    assert usage.cost_usd == 0.5


def test_usage_of_falls_back_to_the_rate_card_for_anthropic_models():
    usage = usage_of("anthropic/claude-sonnet-5", _completion("t", cost=None))
    assert usage.cost_usd is not None and usage.cost_usd > 0
    assert usage_of("openai/gpt-x", _completion("t", cost=None)).cost_usd is None


def test_add_usage_keeps_cost_unknown_once_any_row_is_unpriced():
    priced = Usage(model="m", input_tokens=1, cost_usd=0.1)
    unpriced = Usage(model="m", input_tokens=2, cost_usd=None)
    assert add_usage(priced, priced).cost_usd == pytest.approx(0.2)
    assert add_usage(priced, unpriced).cost_usd is None
    assert add_usage(priced, unpriced).input_tokens == 3


def test_question_text_renders_questions_and_tolerates_bad_json():
    assert question_text('{"questions": [{"question": "Why?"}]}') == "Why?"
    assert question_text("not json") == "not json"


def test_stubbed_tools_stay_within_the_sessions_tool_surface():
    """Every tool the stub table answers by name is one the model is given;
    a stub for a tool that is not offered would never fire."""
    from .generation import DELEGATION_TOOLS, HANDOFF_TOOL, LIBRARY_SEARCH_TOOLS

    offered = {t["function"]["name"] for t in expert_tools(roster_experts(["Max"])[0])}
    named = {
        *LIBRARY_SEARCH_TOOLS,
        *DELEGATION_TOOLS,
        HANDOFF_TOOL,
        "run_agent",
        "view_agent_output",
        "memory_search",
        "list_team",
        "list_schedules",
        "list_presets",
        "list_agent_triggers",
        "list_workspace_files",
    }
    assert named <= offered


def test_expert_session_loses_staffing_tools_and_keeps_memory():
    names = {t["function"]["name"] for t in expert_tools(roster_experts(["Max"])[0])}
    assert "memory_search" in names
    assert "hire_expert" not in names
    assert "update_expert_soul" in names
    plain = {t["function"]["name"] for t in expert_tools(None)}
    assert "hire_expert" in plain
    assert "update_expert_soul" not in plain
