"""Unit tests for the Parallel Decisions blocks beyond the registry smoke test."""

import json
from typing import Any, ClassVar

import pytest

from backend.blocks.parallel import _api
from backend.blocks.parallel._api import DECISION_REQUEST_COST_USD
from backend.blocks.parallel._base import SINGLE_QUESTION
from backend.blocks.parallel._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.parallel._test import (
    CHOICE_ANSWER,
    NOUL_ANSWER,
    REFUSAL_ANSWER,
    SCORE_ANSWER,
    TEST_CONTEXT,
    TEST_LEVELS,
    TEST_OPTIONS,
    mock_response,
)
from backend.blocks.parallel.ask_many import ParallelAskManyDecisionsBlock
from backend.blocks.parallel.choice import ParallelChoiceDecisionBlock
from backend.blocks.parallel.score import ParallelScoreDecisionBlock
from backend.blocks.parallel.yes_no import ParallelYesNoDecisionBlock


class FakeDecisions:
    """Stands in for the API call and records what each block sent."""

    def __init__(self, answers: dict[str, dict[str, Any]]):
        self.answers = answers
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, credentials, state, questions, include_excerpts):
        self.calls.append(
            {
                "state": state,
                "questions": questions,
                "include_excerpts": include_excerpts,
            }
        )
        return mock_response(self.answers)


async def run_block(block, fake: FakeDecisions, **inputs) -> dict[str, Any]:
    block.call_decisions = fake
    input_data = block.Input(
        credentials={
            "provider": "parallel",
            "id": TEST_CREDENTIALS.id,
            "type": "api_key",
            "title": TEST_CREDENTIALS.title,
        },
        **inputs,
    )
    return {
        name: value
        async for name, value in block.run(input_data, credentials=TEST_CREDENTIALS)
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "probability, min_probability, pin",
    [
        (0.9, 0.5, "yes"),
        (0.1, 0.5, "no"),
        (0.5, 0.5, "yes"),
        (0.7, 0.8, "unsure"),
        (0.15, 0.8, "no"),
        # 1 - 0.67 is 0.32999... in floats; the boundary must still count as no.
        (0.33, 0.67, "no"),
        (0.67, 0.67, "yes"),
    ],
)
async def test_yes_no_routes_data_by_probability(probability, min_probability, pin):
    fake = FakeDecisions({SINGLE_QUESTION: {"type": "noul", "noul": probability}})
    outputs = await run_block(
        ParallelYesNoDecisionBlock(),
        fake,
        question="Is it?",
        data="payload",
        min_probability=min_probability,
    )
    assert outputs[pin] == "payload"
    assert outputs["probability"] == probability
    assert "refused" not in outputs


@pytest.mark.asyncio
async def test_yes_no_refusal_goes_to_unsure_not_no():
    fake = FakeDecisions({SINGLE_QUESTION: REFUSAL_ANSWER})
    outputs = await run_block(
        ParallelYesNoDecisionBlock(), fake, question="Is it?", data="payload"
    )
    assert outputs["refused"] is True
    assert outputs["unsure"] == "payload"
    assert "no" not in outputs and "probability" not in outputs


@pytest.mark.asyncio
async def test_missing_context_is_sent_as_empty_object():
    fake = FakeDecisions({SINGLE_QUESTION: NOUL_ANSWER})
    await run_block(ParallelYesNoDecisionBlock(), fake, question="Is it?")
    assert fake.calls[0]["state"] == {}
    assert fake.calls[0]["include_excerpts"] is True


@pytest.mark.asyncio
async def test_choice_sends_options_and_handles_refusal():
    block = ParallelChoiceDecisionBlock()
    fake = FakeDecisions({SINGLE_QUESTION: CHOICE_ANSWER})
    outputs = await run_block(
        block,
        fake,
        question="Which?",
        options=TEST_OPTIONS,
        context=TEST_CONTEXT,
        include_excerpts=False,
    )
    assert fake.calls[0] == {
        "state": TEST_CONTEXT,
        "questions": {
            SINGLE_QUESTION: {
                "type": "choice",
                "instructions": "Which?",
                "criteria": TEST_OPTIONS,
            }
        },
        "include_excerpts": False,
    }
    assert outputs["choice"] == "documented"

    refused = await run_block(
        block,
        FakeDecisions({SINGLE_QUESTION: REFUSAL_ANSWER}),
        question="Which?",
        options=TEST_OPTIONS,
    )
    assert refused["refused"] is True
    assert "choice" not in refused


@pytest.mark.asyncio
async def test_score_handles_refusal_without_a_zero_score():
    outputs = await run_block(
        ParallelScoreDecisionBlock(),
        FakeDecisions({SINGLE_QUESTION: REFUSAL_ANSWER}),
        question="How good?",
        levels=TEST_LEVELS,
    )
    assert outputs["refused"] is True
    assert "score" not in outputs


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["bad name", "trailing\n", ""])
async def test_ask_many_rejects_names_the_api_would_reject(name):
    fake = FakeDecisions({})
    with pytest.raises(ValueError, match="should match pattern"):
        await run_block(
            ParallelAskManyDecisionsBlock(),
            fake,
            questions={name: {"type": "noul", "question": "Is it?"}},
        )
    assert fake.calls == []


def test_ask_many_bad_names_fail_the_pre_run_schema_check():
    # The executor checks inputs against the JSON schema before charging the
    # run; the model validator only runs after the charge.
    error = ParallelAskManyDecisionsBlock.Input.validate_data(
        {
            "credentials": TEST_CREDENTIALS_INPUT,
            "questions": {"bad name": {"type": "noul", "question": "Is it?"}},
        }
    )
    assert error is not None and "bad name" in error


@pytest.mark.asyncio
async def test_ask_many_maps_every_question_type():
    fake = FakeDecisions({"a": NOUL_ANSWER, "b": CHOICE_ANSWER, "c": SCORE_ANSWER})
    outputs = await run_block(
        ParallelAskManyDecisionsBlock(),
        fake,
        questions={
            "a": {
                "type": "noul",
                "question": "A?",
                "criteria": {"true": "T", "false": "F"},
            },
            "b": {"type": "choice", "question": "B?", "options": TEST_OPTIONS},
            "c": {"type": "score", "question": "C?", "levels": TEST_LEVELS},
        },
    )
    assert fake.calls[0]["questions"] == {
        "a": {
            "type": "noul",
            "instructions": "A?",
            "criteria": {"true": "T", "false": "F"},
        },
        "b": {"type": "choice", "instructions": "B?", "criteria": TEST_OPTIONS},
        "c": {"type": "score", "instructions": "C?", "criteria": TEST_LEVELS},
    }
    assert outputs["refused_questions"] == []


@pytest.mark.asyncio
async def test_each_request_records_its_usd_cost():
    block = ParallelYesNoDecisionBlock()
    await run_block(block, FakeDecisions({SINGLE_QUESTION: NOUL_ANSWER}), question="?")
    assert block.execution_stats.provider_cost == DECISION_REQUEST_COST_USD
    assert block.execution_stats.provider_cost_type == "cost_usd"


class FakeResponse:
    def __init__(self, status: int, content: bytes):
        self.status = status
        self.content = content
        self.ok = 200 <= status < 300

    def text(self) -> str:
        return self.content.decode()

    def json(self) -> dict[str, Any]:
        return json.loads(self.content)


class FakeRequests:
    response: ClassVar[FakeResponse]
    sent: ClassVar[dict[str, Any]] = {}

    def __init__(self, **kwargs):
        FakeRequests.sent = {"init": kwargs}

    async def post(self, url, json, **kwargs):
        FakeRequests.sent.update(url=url, json=json, kwargs=kwargs)
        return FakeRequests.response


@pytest.mark.asyncio
async def test_make_decision_sends_the_documented_body(monkeypatch):
    monkeypatch.setattr(_api, "Requests", FakeRequests)
    body = mock_response({"a": NOUL_ANSWER})
    FakeRequests.response = FakeResponse(200, json.dumps(body).encode())
    questions = {"a": {"type": "noul", "instructions": "A?"}}
    assert (
        await _api.make_decision(TEST_CREDENTIALS, TEST_CONTEXT, questions, False)
        == body
    )
    assert FakeRequests.sent["json"] == {
        "state": TEST_CONTEXT,
        "questions": questions,
        "include_excerpts": False,
    }
    # The key travels in x-api-key, which redirects would carry to any host.
    assert FakeRequests.sent["kwargs"] == {"allow_redirects": False}
    assert FakeRequests.sent["init"]["retry_max_attempts"] == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content, message",
    [
        (b'{"type":"error","error":{"ref_id":"r","message":"Bad key"}}', "Bad key"),
        (b'{"code":16,"message":"Invalid API key (C.1)"}', "Invalid API key"),
        (b'{"detail":[{"msg":"field required"}]}', "field required"),
        (b"upstream timeout", "upstream timeout"),
    ],
)
async def test_http_errors_raise_with_the_api_message(monkeypatch, content, message):
    monkeypatch.setattr(_api, "Requests", FakeRequests)
    FakeRequests.response = FakeResponse(401, content)
    with pytest.raises(ValueError, match=f"HTTP 401: .*{message}"):
        await _api.make_decision(TEST_CREDENTIALS, {}, {}, True)
    assert FakeRequests.sent["url"] == "https://api.parallel.ai/v1/decisions"
    assert FakeRequests.sent["init"]["extra_headers"] == {
        "x-api-key": "mock-parallel-api-key"
    }
