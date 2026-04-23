from typing import Any

from firecrawl import FirecrawlApp

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

from ._config import firecrawl


class FirecrawlExtractBlock(Block):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = firecrawl.credentials_field()
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
            id="d1774756-4d9e-40e6-bab1-47ec0ccd81b2",
            description="Firecrawl crawls websites to extract comprehensive data while bypassing blockers.",
            categories={BlockCategory.SEARCH},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        app = FirecrawlApp(api_key=credentials.api_key.get_secret_value())

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

        # Firecrawl surfaces actual credit spend on extract responses
        # (credits_used). 1 Firecrawl credit ≈ $0.001.
        credits_used = getattr(extract_result, "credits_used", None) or 0
        self.merge_stats(
            NodeExecutionStats(
                provider_cost=credits_used * 0.001,
                provider_cost_type="cost_usd",
            )
        )
        yield "data", extract_result.data
