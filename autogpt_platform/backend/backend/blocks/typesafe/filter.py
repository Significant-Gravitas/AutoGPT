from typing import Any

from pydantic import model_validator
from typesafe_sdk import Score

from backend.blocks._base import BlockCategory, BlockOutput
from backend.data.model import SchemaField

from ._base import JevBlockBase
from ._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT, TypeSafeCredentials
from ._schemas import CALIBRATION, JevInput, JevOutput
from ._test import SCORE_ANSWER, TEST_LEVELS, TEST_TRANSPARENCY, mock_result


class JevFilterBlock(JevBlockBase):
    class Input(JevInput):
        items: list[Any] = SchemaField(
            description="Items to score, one sequential API call per item; oversized lists are rejected."
        )
        question: str = SchemaField(
            description="Plain-language question to score for each item, using state as context.",
            min_length=1,
        )
        levels: list[str] = SchemaField(
            description=f"Ordered level descriptions, lowest first. {CALIBRATION}",
            min_length=2,
        )
        min_score: float = SchemaField(
            description="Keep items with returned score at or above this value.", ge=0
        )
        max_items: int = SchemaField(
            description="Maximum number of items allowed in this execution.",
            default=100,
            ge=1,
            le=1000,
        )

        @model_validator(mode="after")
        def validate_bounds(self):
            if len(self.items) > self.max_items:
                raise ValueError(
                    "items exceeds max_items; split the list before filtering"
                )
            if self.min_score > len(self.levels) - 1:
                raise ValueError("min_score exceeds the highest configured score level")
            return self

    class Output(JevOutput):
        item_index: int = SchemaField(
            description="Zero-based item index emitted before that item's transparency outputs."
        )
        passed: list[Any] = SchemaField(
            description="Items meeting min_score, in original order."
        )
        rejected: list[Any] = SchemaField(
            description="Items below min_score, in original order."
        )
        scores: list[float] = SchemaField(
            description="Returned scores aligned with the complete input items list."
        )

    def __init__(self):
        super().__init__(
            id="76d7ffff-48de-4bbe-a890-d2ced69b1be3",
            description=f"Filter items with sequential Jev scores, one call per item. Each call receives JSON context and item and emits its own transparency outputs. {CALIBRATION}",
            categories={BlockCategory.AI, BlockCategory.LOGIC},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "state": "test",
                "question": "How strong is the evidence?",
                "levels": TEST_LEVELS,
                "items": ["evidence"],
                "min_score": 0.5,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("item_index", 0)]
            + TEST_TRANSPARENCY
            + [("passed", ["evidence"]), ("rejected", []), ("scores", [0.8])],
            test_mock={
                "call_jev": lambda *args, **kwargs: mock_result(
                    {"judgment": SCORE_ANSWER}
                )
            },
        )

    async def run(
        self, input_data: Input, *, credentials: TypeSafeCredentials, **kwargs
    ) -> BlockOutput:
        scores = []
        for index, item in enumerate(input_data.items):
            result = await self.call_jev(
                credentials,
                {"item": item, "context": input_data.state},
                {
                    "judgment": Score(
                        instructions=input_data.question, criteria=input_data.levels
                    )
                },
            )
            yield "item_index", index
            for output in self.transparency(result):
                yield output
            if result.error:
                return
            scores.append(result.answers["judgment"]["score"])
        yield "passed", [
            item
            for item, score in zip(input_data.items, scores)
            if score >= input_data.min_score
        ]
        yield "rejected", [
            item
            for item, score in zip(input_data.items, scores)
            if score < input_data.min_score
        ]
        yield "scores", scores
