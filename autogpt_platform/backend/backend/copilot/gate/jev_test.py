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

from backend.blocks.typesafe._client import JevCallResult
from backend.copilot.gate import jev
from backend.copilot.gate.classifier import supervise

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
        "A check flagged this call on rubric question 1 "
        "(going beyond the request). Say where."
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


async def test_without_a_key_jev_is_never_called():
    with patch.object(jev, "_api_key", ""):
        judgement, llm, call_jev = await _supervise(
            _jev("allow"), "ask\nreason: posts out", enabled=None
        )

    assert not judgement.allowed
    assert judgement.decided_by == "llm"
    call_jev.assert_not_awaited()
    llm.assert_awaited_once()


async def _supervise(call_jev, llm_answer, *, enabled: bool | None = True):
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
                judgement = await _judge()
        else:
            judgement = await _judge()
    return judgement, llm, call_jev


async def _judge():
    return await supervise(
        tool_name="bash_exec",
        args={"command": "ls reports/"},
        user_message="list the files in reports",
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
