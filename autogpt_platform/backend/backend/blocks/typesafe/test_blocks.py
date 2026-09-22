from unittest.mock import AsyncMock

import pytest

from ._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from ._test import CHOICE_ANSWER, SCORE_ANSWER, TEST_LEVELS, TEST_OPTIONS, mock_result
from .ask_many import JevAskManyBlock
from .choice import JevChoiceBlock
from .filter import JevFilterBlock
from .pick_best import CANDIDATE_DESCRIPTION_LIMIT, JevPickBestBlock
from .route import JevRouteBlock
from .score import JevScoreBlock
from .yes_no import JevYesNoBlock

BLOCKS = (
    JevChoiceBlock,
    JevScoreBlock,
    JevAskManyBlock,
    JevRouteBlock,
    JevYesNoBlock,
    JevPickBestBlock,
    JevFilterBlock,
)
COMMON = {"credentials": TEST_CREDENTIALS_INPUT, "state": {"context": "evidence"}}


@pytest.mark.parametrize("block_type", BLOCKS)
async def test_embedded_registry_cases_are_mocked(block_type, monkeypatch):
    block = block_type()
    mock = AsyncMock(side_effect=block.test_mock["call_jev"])
    monkeypatch.setattr(block, "call_jev", mock)
    inputs = block.input_schema.model_validate(block.test_input)
    outputs = [
        output async for output in block.run(inputs, credentials=TEST_CREDENTIALS)
    ]
    assert outputs == block.test_output
    mock.assert_awaited_once()


async def test_choice_preserves_judgment_and_state(monkeypatch):
    block = JevChoiceBlock()
    answer = {**CHOICE_ANSWER, "confidence": 0.37}
    mock = AsyncMock(return_value=mock_result({"judgment": answer}))
    monkeypatch.setattr(block, "call_jev", mock)
    inputs = block.Input(**COMMON, question="Evidence?", options=TEST_OPTIONS)
    output = dict(
        [item async for item in block.run(inputs, credentials=TEST_CREDENTIALS)]
    )
    assert output["choice"] == answer["choice"]
    assert output["probabilities"] == answer["probabilities"]
    assert output["confidence"] == 0.37
    assert mock.call_args.args[1] == COMMON["state"]
    question = mock.call_args.args[2]["judgment"]
    assert question.instructions == "Evidence?"
    assert question.criteria == TEST_OPTIONS


async def test_score_returns_fractional_score_without_calibration_heuristics(
    monkeypatch,
):
    block = JevScoreBlock()
    answer = {**SCORE_ANSWER, "score": 0.73}
    mock = AsyncMock(return_value=mock_result({"judgment": answer}))
    monkeypatch.setattr(block, "call_jev", mock)
    inputs = block.Input(**COMMON, question="Evidence?", levels=TEST_LEVELS)
    output = dict(
        [item async for item in block.run(inputs, credentials=TEST_CREDENTIALS)]
    )
    assert output["score"] == 0.73
    assert output["max_score"] == 1
    assert output["legend"] == answer["legend"]
    assert mock.call_args.args[2]["judgment"].criteria == TEST_LEVELS


async def test_ask_many_shares_one_call_and_supports_noul(monkeypatch):
    block = JevAskManyBlock()
    answers = {
        "choice": CHOICE_ANSWER,
        "score": SCORE_ANSWER,
        "noul": {"type": "noul", "noul": 0.65},
    }
    mock = AsyncMock(return_value=mock_result(answers))
    monkeypatch.setattr(block, "call_jev", mock)
    questions = {
        "choice": {"type": "choice", "question": "Evidence?", "options": TEST_OPTIONS},
        "score": {
            "type": "score",
            "question": "Evidence strength?",
            "levels": TEST_LEVELS,
        },
        "noul": {
            "type": "noul",
            "question": "Evidence present?",
            "criteria": {"true": "Present", "false": "Absent"},
        },
    }
    inputs = block.Input.model_validate({**COMMON, "questions": questions})
    output = dict(
        [item async for item in block.run(inputs, credentials=TEST_CREDENTIALS)]
    )
    assert output["answers"] == answers
    mock.assert_awaited_once()
    sent = mock.call_args.args[2]
    assert list(sent) == list(questions)
    assert sent["noul"].criteria == {"true": "Present", "false": "Absent"}


@pytest.mark.parametrize("winner", range(5))
async def test_route_only_fires_winning_pin_in_dictionary_order(winner, monkeypatch):
    block = JevRouteBlock()
    options = {name: name for name in ("z", "a", "q", "b", "r")}
    answer = {**CHOICE_ANSWER, "choice": list(options)[winner]}
    monkeypatch.setattr(
        block, "call_jev", AsyncMock(return_value=mock_result({"judgment": answer}))
    )
    data = {"original": [1, 2]}
    inputs = block.Input(**COMMON, question="Route?", options=options, data=data)
    output = dict(
        [item async for item in block.run(inputs, credentials=TEST_CREDENTIALS)]
    )
    branches = {
        key: value for key, value in output.items() if key.startswith("option_")
    }
    assert branches == {f"option_{winner + 1}": data}


@pytest.mark.parametrize(
    "choice,confidence,threshold,pin",
    [
        ("yes", 0.8, 0.8, "yes"),
        ("no", 0.8, 0.0, "no"),
        ("yes", 0.79, 0.8, "unsure"),
        ("no", 0.0, 0.0, "no"),
    ],
)
async def test_yes_no_uses_exact_confidence_threshold(
    choice, confidence, threshold, pin, monkeypatch
):
    block = JevYesNoBlock()
    answer = {**CHOICE_ANSWER, "choice": choice, "confidence": confidence}
    monkeypatch.setattr(
        block, "call_jev", AsyncMock(return_value=mock_result({"judgment": answer}))
    )
    inputs = block.Input(
        **COMMON, question="Evidence?", data=None, min_confidence=threshold
    )
    output = dict(
        [item async for item in block.run(inputs, credentials=TEST_CREDENTIALS)]
    )
    assert {
        key: value for key, value in output.items() if key in {"yes", "no", "unsure"}
    } == {pin: None}
    assert output["confidence"] == confidence


async def test_pick_best_uses_choice_for_winner_and_probability_for_ranking(
    monkeypatch,
):
    block = JevPickBestBlock()
    candidates = [{"id": 1}, {"id": 2}, {"id": 3}]
    answer = {
        "choice": "candidate_2",
        "probabilities": {"candidate_3": 0.5, "candidate_2": 0.25, "candidate_1": 0.25},
    }
    mock = AsyncMock(return_value=mock_result({"judgment": answer}))
    monkeypatch.setattr(block, "call_jev", mock)
    inputs = block.Input(**COMMON, question="Best?", candidates=candidates)
    output = dict(
        [item async for item in block.run(inputs, credentials=TEST_CREDENTIALS)]
    )
    assert output["best"] == candidates[1]
    assert output["best_index"] == 1
    assert output["ranked"] == [candidates[2], candidates[0], candidates[1]]
    assert output["probabilities"] == answer["probabilities"]
    assert mock.call_args.args[2]["judgment"].criteria["candidate_1"] == '{"id":1}'


async def test_pick_best_reports_description_truncation(monkeypatch):
    block = JevPickBestBlock()
    answer = {
        "choice": "candidate_1",
        "probabilities": {"candidate_1": 0.8, "candidate_2": 0.2},
    }
    mock = AsyncMock(return_value=mock_result({"judgment": answer}))
    monkeypatch.setattr(block, "call_jev", mock)
    inputs = block.Input(**COMMON, question="Best?", candidates=["x" * 3000, "short"])
    output = dict(
        [item async for item in block.run(inputs, credentials=TEST_CREDENTIALS)]
    )
    assert (
        len(mock.call_args.args[2]["judgment"].criteria["candidate_1"])
        == CANDIDATE_DESCRIPTION_LIMIT
    )
    assert output["truncated"] is True
    assert "candidate_1" in output["truncation_note"]
    assert output["best"] == "x" * 3000


async def test_filter_preserves_order_alignment_and_per_call_transparency(monkeypatch):
    block = JevFilterBlock()
    results = [
        mock_result({"judgment": {**SCORE_ANSWER, "score": score}})
        for score in (0.2, 0.5, 0.9)
    ]
    for index, result in enumerate(results):
        result.request_id = f"request-{index}"
    mock = AsyncMock(side_effect=results)
    monkeypatch.setattr(block, "call_jev", mock)
    inputs = block.Input(
        **COMMON,
        items=["low", "equal", "high"],
        question="Quality?",
        levels=TEST_LEVELS,
        min_score=0.5,
    )
    outputs = [item async for item in block.run(inputs, credentials=TEST_CREDENTIALS)]
    output = dict(outputs)
    assert output["passed"] == ["equal", "high"]
    assert output["rejected"] == ["low"]
    assert output["scores"] == [0.2, 0.5, 0.9]
    assert [value for pin, value in outputs if pin == "item_index"] == [0, 1, 2]
    assert [value for pin, value in outputs if pin == "request_id"] == [
        "request-0",
        "request-1",
        "request-2",
    ]
    assert [call.args[1] for call in mock.call_args_list] == [
        {"item": item, "context": COMMON["state"]} for item in inputs.items
    ]
    assert list(mock.call_args_list[0].args[1]) == ["item", "context"]


async def test_filter_empty_input_makes_no_calls(monkeypatch):
    block = JevFilterBlock()
    mock = AsyncMock()
    monkeypatch.setattr(block, "call_jev", mock)
    inputs = block.Input(
        **COMMON, items=[], question="Quality?", levels=TEST_LEVELS, min_score=0.5
    )
    outputs = [item async for item in block.run(inputs, credentials=TEST_CREDENTIALS)]
    assert outputs == [("passed", []), ("rejected", []), ("scores", [])]
    mock.assert_not_awaited()
