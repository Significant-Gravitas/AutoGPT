from typing import Any

from backend.blocks._base import BlockCategory, BlockOutput
from backend.data.model import SchemaField

from ._base import JevBlockBase
from ._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT, TypeSafeCredentials
from ._schemas import CALIBRATION, JevInput, JevOutput, Question
from ._test import (
    CHOICE_ANSWER,
    SCORE_ANSWER,
    TEST_LEVELS,
    TEST_OPTIONS,
    TEST_TRANSPARENCY,
    mock_result,
)


class JevAskManyBlock(JevBlockBase):
    class Input(JevInput):
        questions: dict[str, Question] = SchemaField(
            description=(
                "Named typed judgments sharing one state and one API call: "
                "choice uses question/options, score uses question/levels, "
                "noul uses question and optional true/false criteria. " + CALIBRATION
            ),
            min_length=1,
        )

    class Output(JevOutput):
        answers: dict[str, dict[str, Any]] = SchemaField(
            description="Typed answers as plain dictionaries, keyed by question name."
        )

    def __init__(self):
        answers = {"evidence": CHOICE_ANSWER, "quality": SCORE_ANSWER}
        super().__init__(
            id="840fa616-d3a1-4384-855c-657ff9e8a7e6",
            description=f"Ask multiple Choice, Score, or Noul questions of one shared state with Jev in a single call. {CALIBRATION}",
            categories={BlockCategory.AI},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "state": "test",
                "questions": {
                    "evidence": {
                        "type": "choice",
                        "question": "Is evidence present?",
                        "options": TEST_OPTIONS,
                    },
                    "quality": {
                        "type": "score",
                        "question": "How strong is the evidence?",
                        "levels": TEST_LEVELS,
                    },
                },
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=TEST_TRANSPARENCY + [("answers", answers)],
            test_mock={"call_jev": lambda *args, **kwargs: mock_result(answers)},
        )

    async def run(
        self, input_data: Input, *, credentials: TypeSafeCredentials, **kwargs
    ) -> BlockOutput:
        result = await self.call_jev(
            credentials,
            input_data.state,
            {
                name: question.to_question()
                for name, question in input_data.questions.items()
            },
        )
        for output in self.transparency(result):
            yield output
        if result.error:
            return
        yield "answers", result.answers
