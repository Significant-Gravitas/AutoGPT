from typing import Any

from firecrawl import FirecrawlApp

from backend.blocks.fastcrw._api import get_fastcrw_api_url
from backend.data.model import NodeExecutionStats
from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    SchemaField,
)
from backend.util.exceptions import BlockExecutionError

from ._config import fastcrw


class FastCRWExtractBlock(Block):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = fastcrw.credentials_field()
        urls: list[str] = SchemaField(
            description="The URLs to crawl - at least one is required. Wildcards are supported. (/*)"
        )
        prompt: str | None = SchemaField(
            description="The prompt to use for the crawl", default=None, advanced=False
        )
        output_schema: dict | None = SchemaField(
            description="A Json Schema describing the output structure if more rigid structure is desired.",
            default=None,
        )
        enable_web_search: bool = SchemaField(
            description="When true, extraction can follow links outside the specified domain.",
            default=False,
        )

    class Output(BlockSchemaOutput):
        data: dict[str, Any] = SchemaField(description="The result of the crawl")
        error: str = SchemaField(
            description="Error message if the extraction failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="6af7c35b-6cd8-40c1-8d54-8f1cb7027699",
            description="fastCRW crawls websites to extract comprehensive data while bypassing blockers.",
            categories={BlockCategory.SEARCH},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.urls:
            raise ValueError("At least one URL is required.")

        app = FirecrawlApp(
            api_key=credentials.api_key.get_secret_value(),
            api_url=get_fastcrw_api_url(),
        )

        try:
            extract_result = app.extract(
                urls=input_data.urls,
                prompt=input_data.prompt,
                schema=input_data.output_schema,
                enable_web_search=input_data.enable_web_search,
            )
        except Exception as e:
            raise BlockExecutionError(
                message=f"Extract failed: {e}",
                block_name=self.name,
                block_id=self.id,
            ) from e

        # fastCRW mirrors Firecrawl's credit model and surfaces actual credit
        # spend on extract responses (credits_used). 1 credit ~= $0.001.
        credits_used = max(0, getattr(extract_result, "credits_used", None) or 0)
        self.merge_stats(
            NodeExecutionStats(
                provider_cost=credits_used * 0.001,
                provider_cost_type="cost_usd",
            )
        )
        yield "data", extract_result.data
