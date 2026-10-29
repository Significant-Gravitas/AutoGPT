"""Tests for the ExaAnswerBlock request body and structured answers."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from backend.blocks.exa._test import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.exa.answers import AnswerModel, ExaAnswerBlock, build_answer_payload


def _input(**kwargs) -> ExaAnswerBlock.Input:
    return ExaAnswerBlock.Input(
        credentials=TEST_CREDENTIALS_INPUT, query="What is X?", **kwargs
    )


def test_default_payload():
    assert build_answer_payload(_input()) == {"query": "What is X?", "text": True}


def test_optional_fields_are_sent():
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    payload = build_answer_payload(
        _input(
            model=AnswerModel.EXA_PRO,
            output_schema=schema,
            system_prompt="Be brief",
            user_location="gb",
            user_identifier="u-1",
            text=False,
        )
    )
    assert payload == {
        "query": "What is X?",
        "text": False,
        "model": "exa-pro",
        "outputSchema": schema,
        "systemPrompt": "Be brief",
        "userLocation": "GB",
        "userIdentifier": "u-1",
    }


@pytest.mark.asyncio
async def test_structured_answer_yields_object_and_json_text():
    block = ExaAnswerBlock()
    response = {
        "requestId": "req-2",
        "answer": {"x": "42"},
        "citations": [{"id": "c", "url": "https://a.com", "title": "A"}],
        "costDollars": {"total": 0.005},
    }
    with patch.object(block, "call_exa", AsyncMock(return_value=response)):
        outputs = dict(
            [item async for item in block.run(_input(), credentials=TEST_CREDENTIALS)]
        )
    assert outputs["answer_structured"] == {"x": "42"}
    assert json.loads(outputs["answer"]) == {"x": "42"}
    assert outputs["citation"].title == "A"
    assert outputs["cost_dollars"].total == pytest.approx(0.005)


@pytest.mark.asyncio
async def test_plain_answer_has_no_structured_output():
    block = ExaAnswerBlock()
    with patch.object(
        block, "call_exa", AsyncMock(return_value={"answer": "Paris", "citations": []})
    ):
        outputs = dict(
            [item async for item in block.run(_input(), credentials=TEST_CREDENTIALS)]
        )
    assert outputs["answer"] == "Paris"
    assert "answer_structured" not in outputs
    assert "cost_dollars" not in outputs
