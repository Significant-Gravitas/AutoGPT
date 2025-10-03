from typing import Any

from firecrawl import FirecrawlApp

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockCost,
    BlockCostType,
    BlockOutput,
    BlockSchema,
    CredentialsMetaInput,
    SchemaField,
    cost,
)

from ._config import firecrawl


@cost(BlockCost(2, BlockCostType.RUN))
class FirecrawlExtractBlock(Block):
    class Input(BlockSchema):
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

    class Output(BlockSchema):
        data: dict[str, Any] = SchemaField(description="The result of the crawl")

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

        extract_result = app.extract(
            urls=input_data.urls,
            prompt=input_data.prompt,
            schema=input_data.output_schema,
            enable_web_search=input_data.enable_web_search,
        )

        yield "data", extract_result.data
