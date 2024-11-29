from groq._utils._utils import quote

from backend.blocks.jina._auth import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    JinaCredentials,
    JinaCredentialsField,
    JinaCredentialsInput,
)
from backend.blocks.search import GetRequest
from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField


class SearchTheWebBlock(Block, GetRequest):
    class Input(BlockSchema):
        credentials: JinaCredentialsInput = JinaCredentialsField()
        query: str = SchemaField(description="The search query to search the web for")

    class Output(BlockSchema):
        results: str = SchemaField(
            description="The search results including content from top 5 URLs"
        )
        error: str = SchemaField(description="Error message if the search fails")

    def __init__(self):
        super().__init__(
            id="87840993-2053-44b7-8da4-187ad4ee518c",
            description="This block searches the internet for the given search query.",
            categories={BlockCategory.SEARCH},
            input_schema=SearchTheWebBlock.Input,
            output_schema=SearchTheWebBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "query": "Artificial Intelligence",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=("results", "search content"),
            test_mock={"get_request": lambda *args, **kwargs: "search content"},
        )

    def run(
        self, input_data: Input, *, credentials: JinaCredentials, **kwargs
    ) -> BlockOutput:
        # Encode the search query
        encoded_query = quote(input_data.query)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {credentials.api_key.get_secret_value()}",
        }

        # Prepend the Jina Search URL to the encoded query
        jina_search_url = f"https://s.jina.ai/{encoded_query}"
        results = self.get_request(jina_search_url, headers=headers, json=False)

        # Output the search results
        yield "results", results


class ExtractWebsiteContentBlock(Block, GetRequest):
    class Input(BlockSchema):
        credentials: JinaCredentialsInput = JinaCredentialsField()
        url: str = SchemaField(description="The URL to scrape the content from")
        raw_content: bool = SchemaField(
            default=False,
            title="Raw Content",
            description="Whether to do a raw scrape of the content or use Jina-ai Reader to scrape the content",
            advanced=True,
        )

    class Output(BlockSchema):
        content: str = SchemaField(description="The scraped content from the given URL")
        error: str = SchemaField(
            description="Error message if the content cannot be retrieved"
        )

    def __init__(self):
        super().__init__(
            id="436c3984-57fd-4b85-8e9a-459b356883bd",
            description="This block scrapes the content from the given web URL.",
            categories={BlockCategory.SEARCH},
            input_schema=ExtractWebsiteContentBlock.Input,
            output_schema=ExtractWebsiteContentBlock.Output,
            test_input={
                "url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=("content", "scraped content"),
            test_mock={"get_request": lambda *args, **kwargs: "scraped content"},
        )

    def run(
        self, input_data: Input, *, credentials: JinaCredentials, **kwargs
    ) -> BlockOutput:
        if input_data.raw_content:
            url = input_data.url
            headers = {}
        else:
            url = f"https://r.jina.ai/{input_data.url}"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {credentials.api_key.get_secret_value()}",
            }

        content = self.get_request(url, json=False, headers=headers)
        yield "content", content
