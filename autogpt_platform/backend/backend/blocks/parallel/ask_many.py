import re
from typing import Any

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

QUESTION_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


class ParallelAskManyDecisionsBlock(ParallelDecisionBase):
    class Input(ParallelDecisionInput):
        questions: dict[str, Question] = SchemaField(
            description=(
                "1 to 100 named questions, answered in one request. Names may "
                "use letters, numbers, underscores and hyphens. Each question is "
                "noul (yes/no: question, optional true/false criteria), choice "
                "(question, options) or score (question, levels, lowest first). "
                + GROUPING
            ),
            min_length=1,
            max_length=100,
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
        bad_names = [n for n in input_data.questions if not QUESTION_NAME.match(n)]
        if bad_names:
            raise ValueError(
                "Question names may only use letters, numbers, underscores and "
                f"hyphens: {', '.join(bad_names)}"
            )
        result = await self.decide(
            credentials,
            input_data,
            {name: q.to_api() for name, q in input_data.questions.items()},
        )
        for output in self.evidence(result):
            yield output
        yield "answers", result.answers
        yield "refused_questions", [
            name for name, answer in result.answers.items() if is_refusal(answer)
        ]
