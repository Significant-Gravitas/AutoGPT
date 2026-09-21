from collections.abc import Iterator
from typing import Any

from typesafe_sdk import Choice, Noul, Score

from backend.blocks._base import Block
from backend.data.model import NodeExecutionStats

from ._client import JevCallResult, call_jev
from ._config import TypeSafeCredentials


class JevBlockBase(Block):
    async def call_jev(
        self,
        credentials: TypeSafeCredentials,
        state: Any,
        questions: dict[str, Choice | Score | Noul],
    ) -> JevCallResult:
        return await call_jev(
            api_key=credentials.api_key.get_secret_value(),
            state=state,
            questions=questions,
        )

    def transparency(self, result: JevCallResult) -> Iterator[tuple[str, Any]]:
        if result.input_tokens is not None and result.output_tokens is not None:
            self.merge_stats(
                NodeExecutionStats(
                    input_token_count=result.input_tokens,
                    output_token_count=result.output_tokens,
                )
            )
        data = result.model_dump()
        for pin in (
            "request",
            "response",
            "latency_ms",
            "input_tokens",
            "output_tokens",
            "request_id",
            "truncated",
            "truncation_note",
        ):
            yield pin, data[pin]
        if result.error:
            yield "error", result.error
