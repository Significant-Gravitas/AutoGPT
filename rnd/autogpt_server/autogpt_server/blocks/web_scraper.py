import requests
from autogpt_server.data.block import Block, BlockSchema, BlockOutput

class WebScraper(Block):
    class Input(BlockSchema):
        url: str  # The URL to scrape

    class Output(BlockSchema):
        content: str  # The scraped content from the URL

    def __init__(self):
        super().__init__(
            id="a1b2c3d4-5e6f-7g8h-9i0j-k1l2m3n4o5p6",  # Unique ID for the block
            input_schema=WebScraper.Input,
            output_schema=WebScraper.Output,
            test_input={"url": "https://en.wikipedia.org/wiki/Artificial_intelligence"},
            test_output={"content": "Artificial intelligence (AI) is intelligence..."},
        )

    def run(self, input_data: Input) -> BlockOutput:
        try:
            # Prepend the Jina-ai Reader URL to the input URL
            jina_url = f"https://r.jina.ai/{input_data.url}"
            
            # Make the request to Jina-ai Reader
            response = requests.get(jina_url)
            response.raise_for_status()
            
            # Output the scraped content
            yield "content", response.text

        except requests.exceptions.HTTPError as http_err:
            raise ValueError(f"HTTP error occurred: {http_err}")
        except requests.RequestException as e:
            raise ValueError(f"Request to Jina-ai Reader failed: {e}")