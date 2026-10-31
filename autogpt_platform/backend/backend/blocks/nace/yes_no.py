from typing import Any

from backend.blocks._base import BlockCategory, BlockEffect, BlockOutput
from backend.data.execution import ExecutionContext
from backend.data.model import SchemaField

from ._base import DrexBlockBase
from ._client import NoulQuestion
from ._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT, NaceCredentials
from ._schemas import DrexInput, DrexOutput
from ._test import NOUL_ANSWER, TEST_TRANSPARENCY, mock_result


class NaceYesNoBlock(DrexBlockBase):
    class Input(DrexInput):
        question: str = SchemaField(
            description="Plain-language question answerable yes or no.", min_length=1
        )
        data: Any = SchemaField(
            default=None,
            advanced=False,
            description="Value forwarded unchanged on yes, no, or unsure.",
        )
        min_confidence: float = SchemaField(
            description="Emit unsure when Drex Max's confidence is below this threshold.",
            default=0.0,
            ge=0.0,
            le=1.0,
        )

    class Output(DrexOutput):
        yes: Any = SchemaField(
            description="Data when Drex Max answers yes and meets the confidence threshold."
        )
        no: Any = SchemaField(
            description="Data when Drex Max answers no and meets the confidence threshold."
        )
        unsure: Any = SchemaField(
            description="Data when confidence is below min_confidence."
        )
        probability: float = SchemaField(
            description="Probability that the answer is yes, from 0 to 1."
        )
        confidence: float = SchemaField(
            description="Confidence in the chosen answer: the larger of the yes and no probabilities."
        )

    def __init__(self):
        super().__init__(
            id="e21c7102-4ba8-43c5-8a88-7bc84b884ece",
            description="Ask Drex Max a plain-language yes/no question about text and up to 8 images, and forward data to the chosen pin, or unsure below your confidence threshold.",
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
            test_output=TEST_TRANSPARENCY
            + [("probability", 0.8), ("confidence", 0.8), ("yes", {"id": 7})],
            test_mock={
                "call_drex": lambda *args, **kwargs: mock_result(
                    {"judgment": NOUL_ANSWER}
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
            {"judgment": NoulQuestion(instructions=input_data.question)},
            images,
        )
        for output in self.transparency(result):
            yield output
        if result.error:
            return
        probability = result.answers["judgment"]["noul"]
        confidence = max(probability, 1 - probability)
        yield "probability", probability
        yield "confidence", confidence
        if confidence < input_data.min_confidence:
            yield "unsure", input_data.data
        else:
            yield "yes" if probability >= 0.5 else "no", input_data.data
