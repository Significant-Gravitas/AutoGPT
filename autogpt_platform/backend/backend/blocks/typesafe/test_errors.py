from unittest.mock import AsyncMock

import pytest

from ._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from ._test import SCORE_ANSWER, TEST_LEVELS, mock_result
from .ask_many import JevAskManyBlock
from .choice import JevChoiceBlock
from .filter import JevFilterBlock
from .pick_best import JevPickBestBlock
from .route import JevRouteBlock
from .score import JevScoreBlock
from .yes_no import JevYesNoBlock


@pytest.mark.parametrize(
    "block_type",
    [
        JevChoiceBlock,
        JevScoreBlock,
        JevAskManyBlock,
        JevRouteBlock,
        JevYesNoBlock,
        JevPickBestBlock,
        JevFilterBlock,
    ],
)
async def test_failed_calls_emit_transparency_and_no_judgment(block_type, monkeypatch):
    block = block_type()
    failure = mock_result({}).model_copy(
        update={
            "error": "Jev API returned HTTP 401",
            "response": '{"error":"unauthorized"}',
            "input_tokens": None,
            "output_tokens": None,
        }
    )
    monkeypatch.setattr(block, "call_jev", AsyncMock(return_value=failure))
    inputs = block.input_schema.model_validate(block.test_input)
    outputs = dict(
        [item async for item in block.run(inputs, credentials=TEST_CREDENTIALS)]
    )
    expected = failure.model_dump(exclude={"answers"})
    if block_type == JevFilterBlock:
        expected["item_index"] = 0
    assert outputs == expected


async def test_filter_failure_keeps_transcripts_without_partial_result_lists(
    monkeypatch,
):
    block = JevFilterBlock()
    success = mock_result({"judgment": SCORE_ANSWER})
    failure = mock_result({}).model_copy(
        update={
            "error": "Jev connection failed",
            "response": None,
            "input_tokens": None,
            "output_tokens": None,
        }
    )
    mock = AsyncMock(side_effect=[success, failure])
    monkeypatch.setattr(block, "call_jev", mock)
    inputs = block.Input.model_validate(
        {
            "credentials": TEST_CREDENTIALS_INPUT,
            "state": "context",
            "items": ["first", "failed", "never"],
            "question": "Quality?",
            "levels": TEST_LEVELS,
            "min_score": 0.5,
        }
    )
    outputs = [item async for item in block.run(inputs, credentials=TEST_CREDENTIALS)]
    assert [value for pin, value in outputs if pin == "item_index"] == [0, 1]
    assert [value for pin, value in outputs if pin == "response"] == [
        success.response,
        None,
    ]
    assert outputs[-1] == ("error", "Jev connection failed")
    assert not {"passed", "rejected", "scores"}.intersection(dict(outputs))
    assert mock.await_count == 2
