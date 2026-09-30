from typesafe_sdk import Score

from backend.blocks._base import BlockCategory, BlockOutput
from backend.data.model import SchemaField

from ._base import JevBlockBase
from ._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT, TypeSafeCredentials
from ._schemas import CALIBRATION, JevInput, JevOutput
from ._test import SCORE_ANSWER, TEST_LEVELS, TEST_TRANSPARENCY, mock_result


class JevScoreBlock(JevBlockBase):
    class Input(JevInput):
        question: str = SchemaField(
            description="Plain-language question to score.", min_length=1
        )
        levels: list[str] = SchemaField(
            description=f"Ordered level descriptions, lowest first. {CALIBRATION}",
            min_length=2,
        )

    class Output(JevOutput):
        score: float = SchemaField(
            description="Score returned by Jev on the 0..max_score scale."
        )
        legend: dict[str, str] = SchemaField(
            description="Level numbers mapped to their descriptions."
        )
        probabilities: dict[str, float] = SchemaField(
            description="Probability for each score level."
        )
        confidence: float = SchemaField(description="Confidence returned by Jev.")
        max_score: int = SchemaField(
            description="Highest possible score: number of levels minus one."
        )

    def __init__(self):
        super().__init__(
            id="8dcc4bd7-09d7-4c52-a70a-d09f5ca044af",
            description=f"Score evidence with Jev using an explicit ordered scale. {CALIBRATION}",
            categories={BlockCategory.AI},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "state": "test",
                "question": "How strong is the evidence?",
                "levels": TEST_LEVELS,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=TEST_TRANSPARENCY
            + [
                (key, SCORE_ANSWER[key])
                for key in ("score", "legend", "probabilities", "confidence")
            ]
            + [("max_score", 1)],
            test_mock={
                "call_jev": lambda *args, **kwargs: mock_result(
                    {"judgment": SCORE_ANSWER}
                )
            },
        )

    async def run(
        self, input_data: Input, *, credentials: TypeSafeCredentials, **kwargs
    ) -> BlockOutput:
        result = await self.call_jev(
            credentials,
            input_data.state,
            {
                "judgment": Score(
                    instructions=input_data.question, criteria=input_data.levels
                )
            },
        )
        for output in self.transparency(result):
            yield output
        if result.error:
            return
        answer = result.answers["judgment"]
        for pin in ("score", "legend", "probabilities", "confidence"):
            yield pin, answer[pin]
        yield "max_score", len(input_data.levels) - 1
