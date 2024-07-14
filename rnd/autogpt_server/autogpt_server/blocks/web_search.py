import requests
from urllib.parse import quote
from autogpt_server.data.block import Block, BlockSchema, BlockOutput

class WebSearch(Block):
    class Input(BlockSchema):
        query: str  # The search query

    class Output(BlockSchema):
        results: str  # The search results including content from top 5 URLs

    def __init__(self):
        super().__init__(
            id="b2c3d4e5-6f7g-8h9i-0j1k-l2m3n4o5p6q7",  # Unique ID for the block
            input_schema=WebSearch.Input,
            output_schema=WebSearch.Output,
            test_input={"query": "Who will win 2024 US presidential election?"},
            test_output={"results": "Search results for 'Who will win 2024 US presidential election?'..."},
        )

    def run(self, input_data: Input) -> BlockOutput:
        try:
            # Encode the search query
            encoded_query = quote(input_data.query)
            
            # Prepend the Jina Search URL to the encoded query
            jina_search_url = f"https://s.jina.ai/{encoded_query}"
            
            # Make the request to Jina Search
            response = requests.get(jina_search_url)
            response.raise_for_status()
            
            # Output the search results
            yield "results", response.text

        except requests.exceptions.HTTPError as http_err:
            raise ValueError(f"HTTP error occurred: {http_err}")
        except requests.RequestException as e:
            raise ValueError(f"Request to Jina Search failed: {e}")