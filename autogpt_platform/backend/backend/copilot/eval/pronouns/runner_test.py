"""Scoring a generated turn, end to end through ``generate_turn`` with a
scripted provider in place of the model."""

import argparse
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai.types.chat import ChatCompletion

from backend.copilot.config import ChatConfig
from backend.copilot.eval.style.generation import Turn, generate_turn
from backend.copilot.eval.style.models import Usage

from .cases import expand_cases, first_message, load_case_file, teammate_expert
from .delegation import delegated_task
from .runner import _positive_int, exit_code, score, summarize

CASE_FILE = load_case_file()
CASES = expand_cases(CASE_FILE)
TEAMMATE = teammate_expert(CASE_FILE.teammate)
OTTO_CASE = next(c for c in CASES if c.kind == "otto_brief")
EXPERT_CASE = next(c for c in CASES if c.kind == "expert_task")


def _completion(text: str | None, call: tuple[str, dict] | None = None):
    message: dict = {"role": "assistant", "content": text}
    if call:
        name, args = call
        message["tool_calls"] = [
            {
                "id": "t1",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        ]
    return ChatCompletion.model_validate(
        {
            "id": "c",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls" if call else "stop",
                    "message": message,
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )


async def _otto_turn(task: str) -> Turn:
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        side_effect=[
            _completion(
                None,
                ("delegate_to_expert", {"expert_id": TEAMMATE.id, "prompt": task}),
            ),
            _completion("Maya has it."),
        ]
    )
    return await generate_turn(
        client,
        ChatConfig(),
        model="anthropic/claude-x",
        expert=None,
        roster=[TEAMMATE],
        user_message=first_message(OTTO_CASE, TEAMMATE),
    )


def _turn(text: str = "", delegated: list[str] | None = None) -> Turn:
    return Turn(
        text=text,
        truncated=False,
        usage=Usage(model="m", cost_usd=0.0),
        tool_calls=[],
        calls=[
            ("delegate_to_expert", json.dumps({"prompt": task}))
            for task in delegated or []
        ],
    )


@pytest.mark.asyncio
async def test_otto_guessing_a_gender_in_the_handoff_fails_the_case():
    turn = await _otto_turn("Priya wants you to run the launch planning for her.")
    result = score(OTTO_CASE, turn)
    assert result.outcome == "fail"
    assert result.pronouns == ["her"]


@pytest.mark.asyncio
async def test_a_handoff_that_uses_the_name_passes():
    turn = await _otto_turn("Priya wants you to run the launch. Ask them for dates.")
    result = score(OTTO_CASE, turn)
    assert result.outcome == "pass"
    assert result.checked == ["Priya wants you to run the launch. Ask them for dates."]


def test_otto_not_delegating_is_inconclusive_not_a_pass():
    assert score(OTTO_CASE, _turn("I'll plan it myself.")).outcome == "no_delegation"


def test_otto_is_scored_on_the_handoff_not_on_its_reply():
    """Otto's reply to the user may say "she" for Maya; only the delegated
    text, where the user is the only third person, is checked."""
    turn = _turn("She's on it.", delegated=["Plan it for Priya."])
    assert score(OTTO_CASE, turn).outcome == "pass"


def test_the_experts_report_is_checked():
    report = "Question for the owner: can he send a photo of the serum?"
    result = score(EXPERT_CASE, _turn(report))
    assert result.outcome == "fail"
    assert result.pronouns == ["he"]
    assert score(EXPERT_CASE, _turn("Priya, can you send a photo?")).outcome == ("pass")


def test_the_delegated_text_is_read_through_the_capability_wrapper():
    args = {"id": "tool:handoff_to_expert", "input": {"prompt": "p", "context": "c"}}
    assert delegated_task("run_capability", json.dumps(args)) == "c\n\np"
    assert delegated_task("delegate_to_expert", '{"prompt": "p"}') == "p"
    assert delegated_task("memory_search", '{"query": "q"}') is None
    assert delegated_task("delegate_to_expert", "not json") is None


def test_the_summary_counts_every_outcome():
    results = [score(EXPERT_CASE, _turn("he")), score(EXPERT_CASE, _turn("they"))]
    assert summarize(results).splitlines()[-1] == (
        "2 runs: 1 pass, 1 fail, 0 no_delegation, 0 error"
    )


def _passing(case) -> object:
    """A clean, scored result for ``case``."""
    if case.kind == "otto_brief":
        return score(case, _turn(delegated=["Plan it for the owner."]))
    return score(case, _turn("Can you send the photo?"))


def test_a_run_passes_only_when_every_case_was_scored_clean():
    results = [_passing(case) for case in CASES]
    assert {r.outcome for r in results} == {"pass"}
    assert exit_code(results) == 0
    failed = score(EXPERT_CASE, _turn("Can he send it?"))
    errored = score(EXPERT_CASE, _turn(""))
    assert exit_code([*results, failed]) == 1
    assert exit_code([*results, errored]) == 1
    assert exit_code([]) == 1


def test_a_persona_that_was_never_scored_fails_the_run():
    """Astra: one pass per kind let the run pass with Sam and the unnamed
    owner never exercised. Every persona and case must be scored."""
    priya_only = [_passing(c) for c in CASES if c.persona.id == "priya-dewleaf"]
    assert {r.kind for r in priya_only} == {"otto_brief", "expert_task"}
    assert exit_code(priya_only) == 1
    assert "sam-harbor-bikes/otto_brief-1" in summarize(priya_only)


def test_a_case_that_never_delegated_leaves_the_run_incomplete():
    results = [_passing(case) for case in CASES]
    undelegated = score(OTTO_CASE, _turn("I'll do it myself."))
    assert undelegated.outcome == "no_delegation"
    rest = [r for r in results if r.case_id != OTTO_CASE.id]
    assert exit_code([*rest, undelegated]) == 1
    # A clean pass on another repeat of the same case completes it.
    assert exit_code([*rest, undelegated, _passing(OTTO_CASE)]) == 0


@pytest.mark.parametrize("value", ["0", "-2"])
def test_concurrency_and_repeats_must_be_positive(value):
    with pytest.raises(argparse.ArgumentTypeError):
        _positive_int(value)
    assert _positive_int("3") == 3


@pytest.mark.parametrize("limit", ["truncated", "hit_round_cap"])
def test_an_unfinished_report_cannot_pass(limit):
    clean = _turn("Priya, can you send the").model_copy(update={limit: True})
    result = score(EXPERT_CASE, clean)
    assert result.outcome == "error"
    assert result.error
    gendered = _turn("Can he send the").model_copy(update={limit: True})
    assert score(EXPERT_CASE, gendered).outcome == "fail"


def test_otto_delegations_still_score_at_the_round_cap_but_not_when_truncated():
    capped = _turn(delegated=["Plan it for Priya."]).model_copy(
        update={"hit_round_cap": True}
    )
    assert score(OTTO_CASE, capped).outcome == "pass"
    truncated = capped.model_copy(update={"truncated": True})
    assert score(OTTO_CASE, truncated).outcome == "error"
