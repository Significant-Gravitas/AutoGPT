from backend.blocks._base import BlockCategory, BlockEffect, BlockOutput
from backend.data.execution import ExecutionContext
from backend.data.model import SchemaField

from ._base import DrexBlockBase
from ._client import ScoreQuestion
from ._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT, NaceCredentials
from ._schemas import CALIBRATION, DrexInput, DrexOutput
from ._test import SCORE_ANSWER, TEST_LEVELS, TEST_TRANSPARENCY, mock_result

MAX_LEVELS = 255


class NaceScoreBlock(DrexBlockBase):
    class Input(DrexInput):
        question: str = SchemaField(
            description="Plain-language question to score.", min_length=1
        )
        levels: list[str] = SchemaField(
            description=f"Ordered level descriptions, lowest first (2-255 levels). {CALIBRATION}",
            min_length=2,
            max_length=MAX_LEVELS,
        )

    class Output(DrexOutput):
        score: float = SchemaField(
            description="Score returned by Drex Max on the 0..max_score scale."
        )
        legend: dict[str, str] = SchemaField(
            description="Level numbers mapped to their descriptions."
        )
        probabilities: dict[str, float] = SchemaField(
            description="Probability for each score level."
        )
        confidence: float = SchemaField(description="Confidence returned by Drex Max.")
        max_score: int = SchemaField(
            description="Highest possible score: number of levels minus one."
        )

    def __init__(self):
        super().__init__(
            id="b13a8511-c600-4e4b-9af3-0b1669947632",
            description=f"Score text and up to 8 images with Drex Max using an explicit ordered scale. {CALIBRATION}",
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
                ("score", 0.8),
                ("legend", {"0": TEST_LEVELS[0], "1": TEST_LEVELS[1]}),
                ("probabilities", SCORE_ANSWER["probabilities"]),
                ("confidence", 0.8),
                ("max_score", 1),
            ],
            test_mock={
                "call_drex": lambda *args, **kwargs: mock_result(
                    {"judgment": SCORE_ANSWER}
                )
            },
            effect=BlockEffect.READ,
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: NaceCredentials,
        execution_context: ExecutionContext,
        **kwargs,
    ) -> BlockOutput:
        images = await self.resolve_images(input_data.images, execution_context)
        result = await self.call_drex(
            credentials,
            input_data.state,
            {
                "judgment": ScoreQuestion(
                    instructions=input_data.question, criteria=list(input_data.levels)
                )
            },
            images,
        )
        for output in self.transparency(result):
            yield output
        if result.error:
            return
        answer = result.answers["judgment"]
        yield "score", answer["score"]
        # nace returns the legend as the ordered level list; number it like Jev does.
        yield "legend", {
            str(index): label for index, label in enumerate(answer["legend"])
        }
        yield "probabilities", answer["probabilities"]
        yield "confidence", answer["confidence"]
        yield "max_score", len(input_data.levels) - 1
