"""Jev decides; the LLM only explains an ask and can never turn it into a run.

The double answers with Jev's own recorded responses (``testdata/jev_answers.json``)
and the failure shapes ``call_jev`` returns, so the decision under test is the
gate's reading of real answers, not a re-implementation of it.
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prometheus_client import REGISTRY

from backend.blocks.typesafe._budget import prepare_state
from backend.blocks.typesafe._client import JevCallResult
from backend.copilot.gate import jev
from backend.copilot.gate.classifier import (
    ACTION_RUBRIC,
    JUDGED_ACTION_RUBRIC,
    supervise,
)
from backend.copilot.gate.judged_rubric import QUESTION_3_NAME

_MOD = "backend.copilot.gate.classifier"
_ANSWERS = json.loads(
    (Path(__file__).parent / "testdata/jev_answers.json").read_text(encoding="utf-8")
)


async def test_a_jev_allow_runs_without_asking_the_llm():
    judgement, llm, _ = await _supervise(_jev("allow"), "ask\nreason: never asked")

    assert judgement.allowed
    assert judgement.decided_by == "jev"
    assert judgement.first_stage and judgement.first_stage["must_ask"] == 0.14
    llm.assert_not_awaited()


async def test_a_jev_ask_takes_the_llm_reason_and_names_the_flagged_question():
    judgement, llm, _ = await _supervise(
        _jev("ask_on_q1"), "ask\nreason: the edit also deletes the old schedule"
    )

    assert not judgement.allowed
    assert judgement.reason == "the edit also deletes the old schedule"
    assert judgement.decided_by == "jev+llm"
    prompt = llm.await_args.kwargs["messages"][1]["content"]
    assert prompt.endswith(
        "A check flagged this call; rubric question 1 (going beyond the request) "
        "is the likeliest (p 0.97). Say where."
    )


@pytest.mark.parametrize(
    "llm_answer",
    ["allow\nreason: looks fine", "gibberish", RuntimeError("provider down")],
)
async def test_a_jev_ask_holds_whatever_the_llm_says(llm_answer):
    judgement, _, _ = await _supervise(_jev("ask_on_q1"), llm_answer)

    assert not judgement.allowed
    assert judgement.reason == (
        "A check flagged this as possibly going beyond the request; "
        "could not pinpoint where."
    )
    assert judgement.decided_by == "jev+llm"


@pytest.mark.parametrize(
    "failure",
    [
        "Jev API request failed (HTTP 429).",
        "Jev API request failed (HTTP 422).",
        "Jev connection failed or timed out; no HTTP response was received.",
    ],
)
async def test_a_jev_failure_falls_through_to_the_llm(failure):
    judgement, llm, _ = await _supervise(
        AsyncMock(return_value=_result({}, error=failure)), "ask\nreason: posts out"
    )

    assert (judgement.allowed, judgement.reason) == (False, "posts out")
    assert judgement.decided_by == "llm"
    llm.assert_awaited_once()


async def test_a_jev_timeout_falls_through_to_the_llm():
    async def hang(*_args, **_kwargs):
        await asyncio.Event().wait()

    with patch.object(jev.config, "gate_jev_timeout_s", 0.01):
        judgement, llm, _ = await _supervise(
            AsyncMock(side_effect=hang), "allow\nreason: lists files"
        )

    assert judgement.allowed
    assert judgement.decided_by == "llm"
    llm.assert_awaited_once()


async def test_an_unreadable_jev_answer_falls_through_to_the_llm():
    answers = {**_ANSWERS["allow"], "must_ask": {"type": "noul"}}
    judgement, llm, _ = await _supervise(
        AsyncMock(return_value=_result(answers)), "ask\nreason: posts out"
    )

    assert not judgement.allowed
    assert judgement.decided_by == "llm"
    llm.assert_awaited_once()


@pytest.mark.parametrize("threshold, allowed", [(None, True), (0.4, False)])
async def test_a_configured_threshold_adds_an_ask_to_an_allow_choice(
    threshold, allowed
):
    answers = {**_ANSWERS["allow"], "must_ask": {"type": "noul", "noul": 0.45}}
    with patch.object(jev.config, "gate_jev_ask_threshold", threshold):
        judgement, _, _ = await _supervise(
            AsyncMock(return_value=_result(answers)), "ask\nreason: posts out"
        )

    assert judgement.allowed is allowed


async def test_without_a_key_jev_is_never_called():
    with patch.object(jev, "_api_key", ""):
        judgement, llm, call_jev = await _supervise(
            _jev("allow"), "ask\nreason: posts out", enabled=None
        )

    assert not judgement.allowed
    assert judgement.decided_by == "llm"
    call_jev.assert_not_awaited()
    llm.assert_awaited_once()


@pytest.mark.parametrize("outward_judged", [False, True])
async def test_a_judged_outward_call_meets_its_own_rubric_in_both_judges(
    outward_judged,
):
    """Item 3 of the measured rubric asks on anything that leaves the platform,
    so a judged call read by it would always ask, on the second look at the
    whole chat as much as the first."""
    llm = AsyncMock(return_value=_response("ask\nreason: posts a key"))
    call_jev = _jev("ask_on_q1")
    with (
        patch(f"{_MOD}.call_provider_openai_compat_sync", llm),
        patch("backend.copilot.service._get_aux_client", MagicMock()),
        patch.object(jev, "call_jev", call_jev),
        patch.object(jev.config, "gate_first_stage", "jev"),
        patch.object(jev, "_api_key", "test-key"),
    ):
        await supervise(
            tool_name="run_capability",
            args={"id": "discord-send"},
            user_messages=_LONG_CHAT,
            outward_judged=outward_judged,
        )
    jev_rubric = jev.JUDGED_JEV_RUBRIC if outward_judged else jev.JEV_RUBRIC
    assert call_jev.await_count == 2
    for call in call_jev.await_args_list:
        state, questions = call.args[1:3]
        assert state.startswith(jev_rubric + "\n\n<<<BEGIN USER REQUEST")
        assert questions is (jev.JUDGED_QUESTIONS if outward_judged else jev.QUESTIONS)
    system = llm.await_args.kwargs["messages"][0]["content"]
    assert system == (JUDGED_ACTION_RUBRIC if outward_judged else ACTION_RUBRIC)


@pytest.mark.parametrize(
    "outward_judged, name",
    [(False, "data leaving the platform"), (True, QUESTION_3_NAME)],
)
async def test_a_judged_call_held_on_question_3_is_named_by_its_own_rubric(
    outward_judged, name
):
    """Leaving the platform is what the judged tier allows, so a judged call's
    card must not give it as the reason."""
    recorded = _ANSWERS["ask_on_q1"]
    on_q3 = {**recorded, "q1": recorded["q3"], "q3": recorded["q1"]}
    llm = AsyncMock(return_value=_response("ask"))
    with (
        patch(f"{_MOD}.call_provider_openai_compat_sync", llm),
        patch("backend.copilot.service._get_aux_client", MagicMock()),
        patch.object(jev, "call_jev", AsyncMock(return_value=_result(on_q3))),
        patch.object(jev.config, "gate_first_stage", "jev"),
        patch.object(jev, "_api_key", "test-key"),
    ):
        judgement = await supervise(
            tool_name="run_capability",
            args={"id": "discord-send"},
            user_messages=["post the notes in #team"],
            outward_judged=outward_judged,
        )
    prompt = llm.await_args.kwargs["messages"][1]["content"]
    assert f"rubric question 3 ({name})" in prompt
    assert judgement.reason == (
        f"A check flagged this as possibly {name}; could not pinpoint where."
    )


# The request scrolled past the first window: four messages came after it.
_LONG_CHAT = [
    "Tidy my workspace: delete the old screenshots under reports/",
    "thanks",
    "what time is it in Tokyo?",
    "and in Lima?",
    "ok go on",
]


async def test_a_not_asked_for_ask_on_a_long_chat_runs_once_the_whole_chat_is_read():
    call_jev = AsyncMock(
        side_effect=[_result(_ANSWERS["ask_on_q1"]), _result(_ANSWERS["allow"])]
    )
    asks = _asks()

    judgement, _, _ = await _supervise(
        call_jev, "ask\nreason: not asked for", user_messages=_LONG_CHAT
    )

    assert judgement.allowed
    first, second = (_request_in(c.args[1]) for c in call_jev.await_args_list)
    assert _LONG_CHAT[0] not in first
    assert all(message in second for message in _LONG_CHAT)
    # The second call is counted like the first.
    assert _asks() == asks + 1


@pytest.mark.parametrize("answer", ["ask_by_choice", "allow"])
async def test_only_a_not_asked_for_ask_is_judged_again(answer):
    call_jev = _jev(answer)

    judgement, _, _ = await _supervise(
        call_jev, "ask\nreason: cannot be undone", user_messages=_LONG_CHAT
    )

    assert judgement.allowed is (answer == "allow")
    call_jev.assert_awaited_once()


async def test_it_is_judged_again_at_most_once():
    call_jev = _jev("ask_on_q1")

    judgement, _, _ = await _supervise(
        call_jev, "ask\nreason: not asked for", user_messages=_LONG_CHAT
    )

    assert not judgement.allowed
    assert call_jev.await_count == 2


async def test_a_chat_the_first_judgement_read_whole_is_not_judged_again():
    call_jev = _jev("ask_on_q1")

    await _supervise(
        call_jev, "ask\nreason: not asked for", user_messages=_LONG_CHAT[-3:]
    )

    call_jev.assert_awaited_once()


async def test_a_history_too_long_to_read_whole_keeps_its_newest_messages():
    notes = [f"note {n}: " + "background " * 1_500 for n in range(8)]
    call_jev = AsyncMock(
        side_effect=[_result(_ANSWERS["ask_on_q1"]), _result(_ANSWERS["allow"])]
    )

    await _supervise(
        call_jev, "ask\nreason: not asked for", user_messages=[*notes, *_LONG_CHAT]
    )

    state = call_jev.await_args_list[1].args[1]
    second = _request_in(state)
    assert not prepare_state(state, jev.QUESTIONS).truncated
    assert all(message in second for message in _LONG_CHAT)
    assert "note 0:" not in second
    assert "omitted by the system" not in second


async def _supervise(
    call_jev,
    llm_answer,
    *,
    enabled: bool | None = True,
    user_messages: list[str] | None = None,
):
    llm = (
        AsyncMock(side_effect=llm_answer)
        if isinstance(llm_answer, BaseException)
        else AsyncMock(return_value=_response(llm_answer))
    )
    with (
        patch(f"{_MOD}.call_provider_openai_compat_sync", llm),
        patch("backend.copilot.service._get_aux_client", MagicMock()),
        patch.object(jev, "call_jev", call_jev),
        patch.object(jev.config, "gate_first_stage", "jev"),
    ):
        if enabled:
            with patch.object(jev, "_api_key", "test-key"):
                judgement = await _judge(user_messages)
        else:
            judgement = await _judge(user_messages)
    return judgement, llm, call_jev


async def _judge(user_messages: list[str] | None = None):
    return await supervise(
        tool_name="bash_exec",
        args={"command": "ls reports/"},
        user_messages=user_messages or ["list the files in reports"],
    )


def _request_in(state: str) -> str:
    return state.split("<<<BEGIN USER REQUEST ")[1].split("<<<END USER REQUEST")[0]


def _asks() -> float:
    return (
        REGISTRY.get_sample_value("copilot_gate_first_stage_total", {"outcome": "ask"})
        or 0.0
    )


def _jev(name: str) -> AsyncMock:
    return AsyncMock(return_value=_result(_ANSWERS[name]))


def _result(answers: dict, error: str = "") -> JevCallResult:
    return JevCallResult(
        answers=answers,
        request="{}",
        response=None if error else json.dumps({"answers": answers}),
        latency_ms=300.0,
        input_tokens=None if error else 1104,
        output_tokens=None if error else 0,
        request_id="" if error else "req-1",
        truncated=False,
        truncation_note="",
        error=error,
    )


def _response(text: str):
    message = MagicMock()
    message.content = text
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response
