from backend.sdk import (
    APIKeyCredentials,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    SchemaField,
)

from ._base import SINGLE_QUESTION, ParallelDecisionBase, is_refusal
from ._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from ._schemas import ParallelDecisionInput, ParallelDecisionOutput, ScoreQuestion
from ._test import SCORE_ANSWER, TEST_CONTEXT, TEST_EVIDENCE, TEST_LEVELS, mock_response


class ParallelScoreDecisionBlock(ParallelDecisionBase):
    class Input(ParallelDecisionInput):
        question: str = SchemaField(
            description="What to rate, e.g. 'How well does Clay's CLI support "
            "company search and enrichment?'",
            min_length=1,
        )
        levels: list[str] = SchemaField(
            description=(
                "2 to 10 ordered level descriptions, lowest first. Describe the "
                "concrete evidence that qualifies for each level."
            ),
            min_length=2,
            max_length=10,
        )

    class Output(ParallelDecisionOutput):
        score: float = SchemaField(
            description=(
                "Probability-weighted level, from 0 to max_score. Can be "
                "fractional, e.g. 1.7."
            )
        )
        legend: dict[str, str] = SchemaField(
            description="Level numbers mapped to their descriptions."
        )
        probabilities: dict[str, float] = SchemaField(
            description="Estimated probability of each level, keyed by level number."
        )
        max_score: int = SchemaField(
            description="Highest possible score: number of levels minus one."
        )
        refused: bool = SchemaField(
            description="True when Parallel declined to answer. Emitted only then."
        )

    def __init__(self):
        super().__init__(
            id="39ccf704-62fb-4f03-b3b2-b4e2f0575e44",
            description=(
                "Rate something on an ordered scale you define, using current web "
                "evidence from Parallel."
            ),
            categories={BlockCategory.SEARCH},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "context": TEST_CONTEXT,
                "question": "How well is Clay's CLI documented?",
                "levels": TEST_LEVELS,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=TEST_EVIDENCE
            + [(key, SCORE_ANSWER[key]) for key in ("score", "legend", "probabilities")]
            + [("max_score", 1)],
            test_mock={
                "call_decisions": lambda *args, **kwargs: mock_response(
                    {SINGLE_QUESTION: SCORE_ANSWER}
                )
            },
            effect=BlockEffect.READ,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        question = ScoreQuestion(
            type="score", question=input_data.question, levels=input_data.levels
        )
        result = await self.decide(
            credentials, input_data, {SINGLE_QUESTION: question.to_api()}
        )
        # Check the answer before any output, so a malformed response
        # fails without first sending partial outputs downstream.
        answer = result.answer(SINGLE_QUESTION, "score", "legend", "probabilities")
        for output in self.evidence(result):
            yield output
        if is_refusal(answer):
            yield "refused", True
            return
        for pin in ("score", "legend", "probabilities"):
            yield pin, answer[pin]
        yield "max_score", len(input_data.levels) - 1
