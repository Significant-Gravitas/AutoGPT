from backend.blocks._base import BlockCategory, BlockEffect, BlockOutput
from backend.data.execution import ExecutionContext
from backend.data.model import SchemaField

from ._base import DrexBlockBase
from ._client import ChoiceQuestion
from ._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT, NaceCredentials
from ._schemas import DrexInput, DrexOutput
from ._test import CHOICE_ANSWER, TEST_OPTIONS, TEST_TRANSPARENCY, mock_result

MAX_OPTIONS = 255


class NaceChoiceBlock(DrexBlockBase):
    class Input(DrexInput):
        question: str = SchemaField(
            description="Plain-language judgment to make.", min_length=1
        )
        options: dict[str, str] = SchemaField(
            description="Option names mapped to concrete descriptions of what qualifies (2-255 options).",
            min_length=2,
            max_length=MAX_OPTIONS,
        )

    class Output(DrexOutput):
        choice: str = SchemaField(
            description="Winning option name returned by Drex Max."
        )
        probabilities: dict[str, float] = SchemaField(
            description="Probability for each option."
        )
        confidence: float = SchemaField(description="Confidence returned by Drex Max.")

    def __init__(self):
        super().__init__(
            id="8e7f2171-cd91-4f9c-9c18-dd328be74e76",
            description="Make a typed choice with Drex Max about text and up to 8 images. Describe what qualifies for each option in plain language.",
            categories={BlockCategory.AI},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "state": "test",
                "question": "Is evidence present?",
                "options": TEST_OPTIONS,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=TEST_TRANSPARENCY
            + [
                (key, CHOICE_ANSWER[key])
                for key in ("choice", "probabilities", "confidence")
            ],
            test_mock={
                "call_drex": lambda *args, **kwargs: mock_result(
                    {"judgment": CHOICE_ANSWER}
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
                "judgment": ChoiceQuestion(
                    instructions=input_data.question,
                    criteria=dict(input_data.options),
                )
            },
            images,
        )
        for output in self.transparency(result):
            yield output
        if result.error:
            return
        answer = result.answers["judgment"]
        for pin in ("choice", "probabilities", "confidence"):
            yield pin, answer[pin]
