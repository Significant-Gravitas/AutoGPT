import requests
from groq._utils._utils import quote

from backend.blocks.jina._auth import (
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
            test_input={"query": "Artificial Intelligence"},
            test_output=("results", "search content"),
            test_mock={"get_request": lambda url, json: "search content"},
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

        response = requests.get(jina_search_url, headers=headers)

        response.raise_for_status()

        # Output the search results
        yield "results", response.text
