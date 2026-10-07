from linkup import LinkupClient, LinkupFetchResponse

from backend.data.model import NodeExecutionStats
from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    SchemaField,
)
from backend.util.exceptions import BlockExecutionError

from ._api import fetch_cost_usd
from ._config import linkup


class LinkupFetchBlock(Block):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = linkup.credentials_field(
            description="The Linkup integration requires an API Key."
        )
        url: str = SchemaField(description="The URL of the web page to fetch")
        render_js: bool = SchemaField(
            description="Render the page's JavaScript before extracting content (slower, needed for client-rendered pages)",
            default=False,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        markdown: str = SchemaField(
            description="The page content as clean markdown, ready for LLMs"
        )
        error: str = SchemaField(
            description="Error message if the fetch failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="bf1f17df-1d44-4b06-9fcc-3d7c9cd05e03",
            description="Fetches a web page using Linkup and returns its content as markdown",
            categories={BlockCategory.SEARCH},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": linkup.get_test_credentials().model_dump(),
                "url": "https://agpt.co",
            },
            test_credentials=linkup.get_test_credentials(),
            test_output=[
                (
                    "markdown",
                    "# AutoGPT\n\nAutoGPT is a platform for building AI agents.",
                ),
            ],
            test_mock={
                "_fetch": lambda *args, **kwargs: LinkupFetchResponse(
                    markdown="# AutoGPT\n\nAutoGPT is a platform for building AI agents.",
                    favicon="",
                )
            },
            effect=BlockEffect.READ,
        )

    async def _fetch(
        self, credentials: APIKeyCredentials, **kwargs
    ) -> LinkupFetchResponse:
        """Private method to call the Linkup API - can be mocked for testing."""
        client = LinkupClient(api_key=credentials.api_key)
        return await client.async_fetch(**kwargs)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        try:
            response = await self._fetch(
                credentials, url=input_data.url, render_js=input_data.render_js
            )
        except Exception as e:
            raise BlockExecutionError(
                message=f"Fetch failed: {e}",
                block_name=self.name,
                block_id=self.id,
            ) from e

        self.merge_stats(
            NodeExecutionStats(
                provider_cost=fetch_cost_usd(input_data.render_js),
                provider_cost_type="cost_usd",
            )
        )

        yield "markdown", response.markdown
