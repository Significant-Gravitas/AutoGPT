from typing import Any

from typesafe_sdk import Choice

from backend.blocks._base import BlockCategory, BlockOutput
from backend.data.model import SchemaField

from ._base import JevBlockBase
from ._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT, TypeSafeCredentials
from ._schemas import JevInput, JevOutput
from ._test import CHOICE_ANSWER, TEST_TRANSPARENCY, mock_result


class JevYesNoBlock(JevBlockBase):
    class Input(JevInput):
        question: str = SchemaField(
            description="Plain-language question answerable yes or no.", min_length=1
        )
        data: Any = SchemaField(
            default=None,
            advanced=False,
            description="Value forwarded unchanged on yes, no, or unsure.",
        )
        min_confidence: float = SchemaField(
            description="Emit unsure when Jev's confidence is below this threshold.",
            default=0.0,
            ge=0.0,
            le=1.0,
        )

    class Output(JevOutput):
        yes: Any = SchemaField(
            description="Data when Jev chooses yes and meets the confidence threshold."
        )
        no: Any = SchemaField(
            description="Data when Jev chooses no and meets the confidence threshold."
        )
        unsure: Any = SchemaField(
            description="Data when confidence is below min_confidence."
        )
        confidence: float = SchemaField(description="Confidence returned by Jev.")

    def __init__(self):
        super().__init__(
            id="71f7d92c-a895-407f-8966-ede134a854b7",
            description="Ask Jev a plain-language yes/no question and forward data to the chosen pin, or unsure below your confidence threshold.",
            categories={BlockCategory.AI, BlockCategory.LOGIC},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "state": "test",
                "question": "Is evidence present?",
                "data": {"id": 7},
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=TEST_TRANSPARENCY + [("confidence", 0.8), ("yes", {"id": 7})],
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
                    instructions=input_data.question,
                    criteria={
                        "yes": "The answer to the question is yes.",
                        "no": "The answer to the question is no.",
                    },
                )
            },
        )
        for output in self.transparency(result):
            yield output
        if result.error:
            return
        answer = result.answers["judgment"]
        yield "confidence", answer["confidence"]
        pin = (
            "unsure"
            if answer["confidence"] < input_data.min_confidence
            else answer["choice"]
        )
        yield pin, input_data.data
