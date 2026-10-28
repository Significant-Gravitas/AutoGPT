from typing import Annotated, Any

from pydantic import StringConstraints

from backend.sdk import (
    APIKeyCredentials,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    SchemaField,
)

from ._base import ParallelDecisionBase, is_refusal
from ._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from ._schemas import GROUPING, ParallelDecisionInput, ParallelDecisionOutput, Question
from ._test import (
    CHOICE_ANSWER,
    REFUSAL_ANSWER,
    TEST_CONTEXT,
    TEST_EVIDENCE,
    TEST_OPTIONS,
    mock_response,
)

# Names the API accepts. Pydantic checks it when the input is built, but
# that is after the run is charged; propertyNames puts it in the JSON schema
# the executor checks before charging.
QUESTION_NAME_PATTERN = r"^[A-Za-z0-9_-]+$"
QuestionName = Annotated[str, StringConstraints(pattern=QUESTION_NAME_PATTERN)]
# The executor's JSON schema check uses Python's re, where $ also matches
# before a final newline; (?!\n) closes that gap (and is a no-op in JS).
SCHEMA_NAME_PATTERN = r"^[A-Za-z0-9_-]+$(?!\n)"

# Fields each answer type must carry, so a malformed answer is an error
# rather than passed on as if it were valid.
ANSWER_FIELDS = {
    "noul": ("noul",),
    "choice": ("choice", "probabilities"),
    "score": ("score", "legend", "probabilities"),
}


class ParallelAskManyDecisionsBlock(ParallelDecisionBase):
    class Input(ParallelDecisionInput):
        questions: dict[QuestionName, Question] = SchemaField(
            description=(
                "1 to 100 named questions, answered in one request. Names may "
                "use letters, numbers, underscores and hyphens. Each question is "
                "noul (yes/no: question, optional true/false criteria), choice "
                "(question, options) or score (question, levels, lowest first). "
                + GROUPING
            ),
            min_length=1,
            max_length=100,
            json_schema_extra={"propertyNames": {"pattern": SCHEMA_NAME_PATTERN}},
        )

    class Output(ParallelDecisionOutput):
        answers: dict[str, dict[str, Any]] = SchemaField(
            description=(
                "Answers keyed by question name. A noul answer has the "
                "probability it is true in 'noul'; a choice answer has 'choice' "
                "and 'probabilities'; a score answer has 'score', 'legend' and "
                "'probabilities'. A refused question has only type 'refusal'."
            )
        )
        refused_questions: list[str] = SchemaField(
            description="Names of the questions Parallel declined to answer."
        )

    def __init__(self):
        answers = {"cli_availability": CHOICE_ANSWER, "has_sdk": REFUSAL_ANSWER}
        super().__init__(
            id="98a270d8-1844-4976-9ec0-fac846137e86",
            description=(
                "Ask several yes/no, choice and score questions about one subject "
                "in a single Parallel request, answered from shared current web "
                "evidence."
            ),
            categories={BlockCategory.SEARCH},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "context": TEST_CONTEXT,
                "questions": {
                    "cli_availability": {
                        "type": "choice",
                        "question": "What is the availability of Clay's CLI?",
                        "options": TEST_OPTIONS,
                    },
                    "has_sdk": {
                        "type": "noul",
                        "question": "Does Clay publish an official Python SDK?",
                    },
                },
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=TEST_EVIDENCE
            + [("answers", answers), ("refused_questions", ["has_sdk"])],
            test_mock={
                "call_decisions": lambda *args, **kwargs: mock_response(answers)
            },
            effect=BlockEffect.READ,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        result = await self.decide(
            credentials,
            input_data,
            {name: q.to_api() for name, q in input_data.questions.items()},
        )
        # Check the answer before any output, so a malformed response
        # fails without first sending partial outputs downstream.
        for name, question in input_data.questions.items():
            result.answer(name, *ANSWER_FIELDS[question.type])
        for output in self.evidence(result):
            yield output
        yield "answers", result.answers
        yield "refused_questions", [
            name for name, answer in result.answers.items() if is_refusal(answer)
        ]
