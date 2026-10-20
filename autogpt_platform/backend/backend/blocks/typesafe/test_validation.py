import pytest
from pydantic import ValidationError

from ._config import TEST_CREDENTIALS_INPUT
from ._test import TEST_LEVELS
from .ask_many import JevAskManyBlock
from .choice import JevChoiceBlock
from .filter import JevFilterBlock
from .pick_best import JevPickBestBlock
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


@pytest.mark.parametrize(
    "block_type,inputs",
    [
        (JevChoiceBlock, {"question": "Pick?", "options": {"one": "Only"}}),
        (JevScoreBlock, {"question": "Rate?", "levels": ["Only"]}),
        (
            JevRouteBlock,
            {
                "question": "Route?",
                "options": {str(i): str(i) for i in range(6)},
                "data": 1,
            },
        ),
        (JevYesNoBlock, {"question": "Yes?", "min_confidence": 1.1, "data": 1}),
        (JevPickBestBlock, {"question": "Best?", "candidates": []}),
        (
            JevFilterBlock,
            {
                "question": "Rate?",
                "levels": TEST_LEVELS,
                "items": [1, 2],
                "min_score": 0.5,
                "max_items": 1,
            },
        ),
        (
            JevFilterBlock,
            {"question": "Rate?", "levels": TEST_LEVELS, "items": [], "min_score": 2},
        ),
        (
            JevAskManyBlock,
            {"questions": {"bad": {"type": "text", "question": "Generate?"}}},
        ),
        (
            JevAskManyBlock,
            {
                "questions": {
                    "bad": {
                        "type": "noul",
                        "question": "True?",
                        "criteria": {"true": "True"},
                    }
                }
            },
        ),
    ],
)
def test_invalid_inputs_are_rejected_before_api_calls(block_type, inputs):
    with pytest.raises(ValidationError):
        block_type.Input(**COMMON, **inputs)
