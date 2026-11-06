import asyncio
from collections.abc import Iterator
from typing import Any

from backend.blocks._base import Block
from backend.data.execution import ExecutionContext
from backend.data.model import NodeExecutionStats
from backend.util.file import MediaFileType, store_media_file

from ._client import DrexCallResult, Question, call_drex
from ._config import NaceCredentials


class DrexBlockBase(Block):
    async def call_drex(
        self,
        credentials: NaceCredentials,
        state: Any,
        questions: dict[str, Question],
        images: list[str],
    ) -> DrexCallResult:
        return await call_drex(
            api_key=credentials.api_key.get_secret_value(),
            state=state,
            questions=questions,
            images=images,
        )

    async def resolve_images(
        self, images: list[MediaFileType], execution_context: ExecutionContext
    ) -> list[str]:
        """Data URLs for nace from data URLs, http(s) URLs and workspace files."""
        resolved = await asyncio.gather(
            *(
                store_media_file(
                    file=image,
                    execution_context=execution_context,
                    return_format="for_external_api",
                )
                for image in images
            )
        )
        return [str(image) for image in resolved]

    def transparency(self, result: DrexCallResult) -> Iterator[tuple[str, Any]]:
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
