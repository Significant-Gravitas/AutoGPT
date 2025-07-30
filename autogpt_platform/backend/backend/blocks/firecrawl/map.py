from firecrawl import FirecrawlApp

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchema,
    CredentialsMetaInput,
    SchemaField,
)

from ._config import firecrawl


class FirecrawlMapWebsiteBlock(Block):

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = firecrawl.credentials_field()

        url: str = SchemaField(description="The website url to map")

    class Output(BlockSchema):
        links: list[str] = SchemaField(description="The links of the website")

    def __init__(self):
        super().__init__(
            id="f0f43e2b-c943-48a0-a7f1-40136ca4d3b9",
            description="Firecrawl maps a website to extract all the links.",
            categories={BlockCategory.SEARCH},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:

        app = FirecrawlApp(api_key=credentials.api_key.get_secret_value())

        # Sync call
        map_result = app.map_url(
            url=input_data.url,
        )

        yield "links", map_result.links
