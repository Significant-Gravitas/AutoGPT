from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from backend.blocks.nace import _base
from backend.data.execution import ExecutionContext
from backend.util.file import MediaFileType

from ._client import ChoiceQuestion, NoulQuestion, ScoreQuestion
from ._client_test import FakeRequests, FakeResponse
from ._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from ._test import (
    CHOICE_ANSWER,
    NOUL_ANSWER,
    SCORE_ANSWER,
    TEST_IMAGE,
    TEST_LEVELS,
    TEST_OPTIONS,
    mock_result,
)
from .choice import NaceChoiceBlock
from .score import NaceScoreBlock
from .yes_no import NaceYesNoBlock

BLOCKS = (NaceChoiceBlock, NaceScoreBlock, NaceYesNoBlock)
COMMON = {"credentials": TEST_CREDENTIALS_INPUT, "state": {"context": "evidence"}}
CONTEXT = ExecutionContext()


async def run_block(block, inputs) -> list[tuple[str, object]]:
    return [
        item
        async for item in block.run(
            inputs, credentials=TEST_CREDENTIALS, execution_context=CONTEXT
        )
    ]


@pytest.mark.parametrize("block_type", BLOCKS)
async def test_embedded_registry_cases_are_mocked(block_type, monkeypatch):
    block = block_type()
    mock = AsyncMock(side_effect=block.test_mock["call_drex"])
    monkeypatch.setattr(block, "call_drex", mock)
    inputs = block.input_schema.model_validate(block.test_input)
    assert await run_block(block, inputs) == block.test_output
    mock.assert_awaited_once()
    assert mock.call_args.args[3] == []


async def test_choice_sends_options_as_criteria_and_preserves_the_answer(monkeypatch):
    block = NaceChoiceBlock()
    answer = {**CHOICE_ANSWER, "confidence": 0.37}
    mock = AsyncMock(return_value=mock_result({"judgment": answer}))
    monkeypatch.setattr(block, "call_drex", mock)
    inputs = block.Input(**COMMON, question="Evidence?", options=TEST_OPTIONS)
    output = dict(await run_block(block, inputs))
    assert output["choice"] == answer["choice"]
    assert output["probabilities"] == answer["probabilities"]
    assert output["confidence"] == 0.37
    assert mock.call_args.args[1] == COMMON["state"]
    assert mock.call_args.args[2] == {
        "judgment": ChoiceQuestion(instructions="Evidence?", criteria=TEST_OPTIONS)
    }


async def test_score_sends_levels_as_criteria_and_numbers_the_legend(monkeypatch):
    block = NaceScoreBlock()
    answer = {**SCORE_ANSWER, "score": 0.73}
    mock = AsyncMock(return_value=mock_result({"judgment": answer}))
    monkeypatch.setattr(block, "call_drex", mock)
    inputs = block.Input(**COMMON, question="Evidence?", levels=TEST_LEVELS)
    output = dict(await run_block(block, inputs))
    assert output["score"] == 0.73
    assert output["max_score"] == 1
    assert output["legend"] == {"0": TEST_LEVELS[0], "1": TEST_LEVELS[1]}
    assert output["probabilities"] == answer["probabilities"]
    assert mock.call_args.args[2] == {
        "judgment": ScoreQuestion(instructions="Evidence?", criteria=TEST_LEVELS)
    }


@pytest.mark.parametrize(
    "probability,threshold,pin,confidence",
    [
        (0.8, 0.8, "yes", 0.8),
        (0.2, 0.0, "no", 0.8),
        (0.5, 0.0, "yes", 0.5),
        (0.79, 0.8, "unsure", 0.79),
        (0.21, 0.8, "unsure", 0.79),
        (0.0, 0.0, "no", 1.0),
    ],
)
async def test_yes_no_asks_a_noul_question_and_uses_an_exact_threshold(
    probability, threshold, pin, confidence, monkeypatch
):
    block = NaceYesNoBlock()
    answer = {**NOUL_ANSWER, "noul": probability}
    mock = AsyncMock(return_value=mock_result({"judgment": answer}))
    monkeypatch.setattr(block, "call_drex", mock)
    inputs = block.Input(
        **COMMON, question="Evidence?", data={"id": 1}, min_confidence=threshold
    )
    output = dict(await run_block(block, inputs))
    assert {
        key: value for key, value in output.items() if key in {"yes", "no", "unsure"}
    } == {pin: {"id": 1}}
    assert output["probability"] == probability
    assert output["confidence"] == pytest.approx(confidence)
    assert mock.call_args.args[2] == {
        "judgment": NoulQuestion(instructions="Evidence?")
    }


@pytest.mark.parametrize("block_type", BLOCKS)
async def test_images_are_resolved_in_order_and_sent_with_the_question(
    block_type, monkeypatch
):
    resolved = {
        "https://example.com/a.png": "data:image/png;base64,QQ==",
        "workspace://file-b": "data:image/jpeg;base64,Qg==",
        TEST_IMAGE: TEST_IMAGE,
    }

    async def fake_store_media_file(file, execution_context, return_format):
        assert execution_context is CONTEXT
        assert return_format == "for_external_api"
        return MediaFileType(resolved[file])

    monkeypatch.setattr(_base, "store_media_file", fake_store_media_file)
    block = block_type()
    mock = AsyncMock(side_effect=block.test_mock["call_drex"])
    monkeypatch.setattr(block, "call_drex", mock)
    inputs = block.input_schema.model_validate(
        {**block.test_input, "images": list(resolved)}
    )
    outputs = await run_block(block, inputs)
    assert mock.call_args.args[3] == list(resolved.values())
    assert outputs == block.test_output


@pytest.mark.parametrize("block_type", BLOCKS)
async def test_failed_calls_emit_transparency_and_no_judgment(block_type, monkeypatch):
    block = block_type()
    failure = mock_result({}).model_copy(
        update={
            "error": "nace API request failed (HTTP 401).",
            "response": '{"error":{"message":"unauthorized"}}',
            "input_tokens": None,
            "output_tokens": None,
        }
    )
    monkeypatch.setattr(block, "call_drex", AsyncMock(return_value=failure))
    inputs = block.input_schema.model_validate(block.test_input)
    assert dict(await run_block(block, inputs)) == failure.model_dump(
        exclude={"answers"}
    )


@pytest.mark.parametrize("block_type", BLOCKS)
async def test_malformed_http_success_emits_error_without_a_decision(
    block_type, monkeypatch
):
    """The real client failure path must reach every block without missing-key errors."""
    FakeRequests(FakeResponse(200, b'{"answers": {}}')).install(monkeypatch)
    block = block_type()
    inputs = block.input_schema.model_validate(block.test_input)

    output = dict(await run_block(block, inputs))

    assert output["error"] == "nace returned an invalid response (HTTP 200)."
    assert output["response"] == '{"answers": {}}'
    assert not {"choice", "score", "yes", "no", "unsure"}.intersection(output)


@pytest.mark.parametrize(
    "block_type,inputs",
    [
        (NaceChoiceBlock, {"question": "Pick?", "options": {"one": "Only"}}),
        (NaceScoreBlock, {"question": "Rate?", "levels": ["Only"]}),
        (NaceYesNoBlock, {"question": "Yes?", "min_confidence": 1.1}),
        (NaceYesNoBlock, {"question": "", "data": 1}),
        (
            NaceChoiceBlock,
            {"question": "Pick?", "options": TEST_OPTIONS, "images": [TEST_IMAGE] * 9},
        ),
    ],
)
def test_invalid_inputs_are_rejected_before_api_calls(block_type, inputs):
    with pytest.raises(ValidationError):
        block_type.Input(**COMMON, **inputs)


async def test_null_state_and_data_survive_platform_execution(monkeypatch):
    block = NaceYesNoBlock()
    mock = AsyncMock(side_effect=block.test_mock["call_drex"])
    monkeypatch.setattr(block, "call_drex", mock)
    inputs = {**block.test_input, "state": None, "data": None}
    outputs = dict(
        [
            item
            async for item in block.execute(
                inputs, credentials=TEST_CREDENTIALS, execution_context=CONTEXT
            )
        ]
    )
    assert outputs["yes"] is None
    assert mock.call_args.args[1] is None
