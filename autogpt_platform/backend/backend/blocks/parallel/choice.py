from backend.sdk import (
    APIKeyCredentials,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    SchemaField,
)

from ._base import SINGLE_QUESTION, ParallelDecisionBase, is_refusal
from ._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from ._schemas import ChoiceQuestion, ParallelDecisionInput, ParallelDecisionOutput
from ._test import (
    CHOICE_ANSWER,
    TEST_CONTEXT,
    TEST_EVIDENCE,
    TEST_OPTIONS,
    mock_response,
)


class ParallelChoiceDecisionBlock(ParallelDecisionBase):
    class Input(ParallelDecisionInput):
        question: str = SchemaField(
            description="What to decide, e.g. 'What is the availability of "
            "Clay's official CLI?'",
            min_length=1,
        )
        options: dict[str, str] = SchemaField(
            description=(
                "2 to 255 option names, each mapped to a description of when it "
                "applies. Add an 'unknown' option if the evidence may not settle it."
            ),
            min_length=2,
            max_length=255,
        )

    class Output(ParallelDecisionOutput):
        choice: str = SchemaField(description="Name of the chosen option.")
        probabilities: dict[str, float] = SchemaField(
            description="Estimated probability of each option."
        )
        refused: bool = SchemaField(
            description="True when Parallel declined to answer. Emitted only then."
        )

    def __init__(self):
        super().__init__(
            id="9569d6f5-2ff0-4694-8296-c6bda4a894fa",
            description=(
                "Pick one of several options using current web evidence from "
                "Parallel, with a probability for each option."
            ),
            categories={BlockCategory.SEARCH, BlockCategory.LOGIC},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "context": TEST_CONTEXT,
                "question": "What is the availability of Clay's official CLI?",
                "options": TEST_OPTIONS,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=TEST_EVIDENCE
            + [(key, CHOICE_ANSWER[key]) for key in ("choice", "probabilities")],
            test_mock={
                "call_decisions": lambda *args, **kwargs: mock_response(
                    {SINGLE_QUESTION: CHOICE_ANSWER}
                )
            },
            effect=BlockEffect.READ,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        question = ChoiceQuestion(
            type="choice", question=input_data.question, options=input_data.options
        )
        result = await self.decide(
            credentials, input_data, {SINGLE_QUESTION: question.to_api()}
        )
        # Check the answer before any output, so a malformed response
        # fails without first sending partial outputs downstream.
        answer = result.answer(SINGLE_QUESTION, "choice", "probabilities")
        for output in self.evidence(result):
            yield output
        if is_refusal(answer):
            yield "refused", True
            return
        yield "choice", answer["choice"]
        yield "probabilities", answer["probabilities"]
