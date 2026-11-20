from math import isclose
from typing import Any

from backend.sdk import (
    APIKeyCredentials,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    SchemaField,
)

from ._base import SINGLE_QUESTION, ParallelDecisionBase, is_refusal
from ._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from ._schemas import NoulQuestion, ParallelDecisionInput, ParallelDecisionOutput
from ._test import NOUL_ANSWER, TEST_CONTEXT, TEST_EVIDENCE, mock_response


class ParallelYesNoDecisionBlock(ParallelDecisionBase):
    class Input(ParallelDecisionInput):
        question: str = SchemaField(
            description="A statement or question that is either true or false, "
            "e.g. 'Does Clay offer an official command-line interface?'",
            min_length=1,
        )
        data: Any = SchemaField(
            default=None,
            advanced=False,
            description="Value forwarded unchanged to the yes, no or unsure output.",
        )
        min_probability: float = SchemaField(
            default=0.5,
            ge=0.5,
            le=1.0,
            description=(
                "How sure the answer must be before data goes to yes or no. "
                "Anything less certain goes to unsure. 0.5 never gives unsure."
            ),
        )

    class Output(ParallelDecisionOutput):
        probability: float = SchemaField(
            description="Estimated probability, from 0 to 1, that the answer is yes."
        )
        yes: Any = SchemaField(description="Data when the answer is yes.")
        no: Any = SchemaField(description="Data when the answer is no.")
        unsure: Any = SchemaField(
            description="Data when the answer is not certain enough, or Parallel "
            "declined to answer."
        )
        refused: bool = SchemaField(
            description="True when Parallel declined to answer. Emitted only then."
        )

    def __init__(self):
        super().__init__(
            id="e09613ea-aaf5-4393-9a2f-634e8c8ad899",
            description=(
                "Answer a yes/no question using current web evidence from "
                "Parallel, and forward data to the yes, no or unsure output."
            ),
            categories={BlockCategory.SEARCH, BlockCategory.LOGIC},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "context": TEST_CONTEXT,
                "question": "Does Clay offer an official command-line interface?",
                "data": {"id": 7},
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=TEST_EVIDENCE + [("probability", 0.9), ("yes", {"id": 7})],
            test_mock={
                "call_decisions": lambda *args, **kwargs: mock_response(
                    {SINGLE_QUESTION: NOUL_ANSWER}
                )
            },
            effect=BlockEffect.READ,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        question = NoulQuestion(type="noul", question=input_data.question)
        result = await self.decide(
            credentials, input_data, {SINGLE_QUESTION: question.to_api()}
        )
        # Check the answer before any output, so a malformed response
        # fails without first sending partial outputs downstream.
        answer = result.answer(SINGLE_QUESTION, "noul")
        for output in self.evidence(result):
            yield output
        if is_refusal(answer):
            yield "refused", True
            yield "unsure", input_data.data
            return
        probability = answer["noul"]
        yield "probability", probability
        yield _pin(probability, input_data.min_probability), input_data.data


def _pin(probability: float, min_probability: float) -> str:
    if probability >= min_probability:
        return "yes"
    # isclose: 1 - p is inexact in floats (1 - 0.33 < 0.67), which would
    # otherwise send a no exactly at the threshold to unsure.
    no_probability = 1 - probability
    if no_probability >= min_probability or isclose(no_probability, min_probability):
        return "no"
    return "unsure"
