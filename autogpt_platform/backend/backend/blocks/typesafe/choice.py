from typesafe_sdk import Choice

from backend.blocks._base import BlockCategory, BlockOutput
from backend.data.model import SchemaField

from ._base import JevBlockBase
from ._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT, TypeSafeCredentials
from ._schemas import JevInput, JevOutput
from ._test import CHOICE_ANSWER, TEST_OPTIONS, TEST_TRANSPARENCY, mock_result


class JevChoiceBlock(JevBlockBase):
    class Input(JevInput):
        question: str = SchemaField(
            description="Plain-language judgment to make.", min_length=1
        )
        options: dict[str, str] = SchemaField(
            description="Option names mapped to concrete descriptions of what qualifies.",
            min_length=2,
        )

    class Output(JevOutput):
        choice: str = SchemaField(description="Winning option name returned by Jev.")
        probabilities: dict[str, float] = SchemaField(
            description="Probability for each option."
        )
        confidence: float = SchemaField(description="Confidence returned by Jev.")

    def __init__(self):
        super().__init__(
            id="d2b65a78-6b22-44a1-90f8-b8fedfdf85af",
            description="Make a typed choice with Jev. Describe what qualifies for each option in plain language.",
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
                "call_jev": lambda *args, **kwargs: mock_result(
                    {"judgment": CHOICE_ANSWER}
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
                "judgment": Choice(
                    instructions=input_data.question, criteria=input_data.options
                )
            },
        )
        for output in self.transparency(result):
            yield output
        if result.error:
            return
        answer = result.answers["judgment"]
        for pin in ("choice", "probabilities", "confidence"):
            yield pin, answer[pin]
