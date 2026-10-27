from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel

from backend.data.model import NodeExecutionStats
from backend.sdk import APIKeyCredentials, Block

from ._api import DECISION_REQUEST_COST_USD, make_decision
from ._schemas import ParallelCitation, ParallelDecisionInput

# Name the single-question blocks give their one question.
SINGLE_QUESTION = "decision"


class DecisionResult(BaseModel):
    answers: dict[str, dict[str, Any]]
    citations: list[ParallelCitation]
    decision_id: str


class ParallelDecisionBase(Block):
    @staticmethod
    async def call_decisions(
        credentials: APIKeyCredentials,
        state: Any,
        questions: dict[str, dict[str, Any]],
        include_excerpts: bool,
    ) -> dict[str, Any]:
        return await make_decision(credentials, state, questions, include_excerpts)

    async def decide(
        self,
        credentials: APIKeyCredentials,
        input_data: ParallelDecisionInput,
        questions: dict[str, dict[str, Any]],
    ) -> DecisionResult:
        # The API requires a state; an empty object stands in for "no context".
        state = {} if input_data.context is None else input_data.context
        response = await self.call_decisions(
            credentials, state, questions, input_data.include_excerpts
        )
        self.merge_stats(
            NodeExecutionStats(
                provider_cost=DECISION_REQUEST_COST_USD,
                provider_cost_type="cost_usd",
            )
        )
        return DecisionResult.model_validate(response)

    @staticmethod
    def evidence(result: DecisionResult) -> Iterator[tuple[str, Any]]:
        yield "citations", result.citations
        yield "decision_id", result.decision_id


def is_refusal(answer: dict[str, Any]) -> bool:
    """A refusal is a valid answer with no value; never read it as false/zero."""
    return answer.get("type") == "refusal"
