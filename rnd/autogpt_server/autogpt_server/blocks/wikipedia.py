import requests
from autogpt_server.data.block import Block, BlockSchema, BlockOutput

class GetWikipediaSummary(Block):
    class Input(BlockSchema):
        topic: str

    class Output(BlockSchema):
        summary: str

    def __init__(self):
        super().__init__(
            id="h5e7f8g9-1b2c-3d4e-5f6g-7h8i9j0k1l2m",
            input_schema=GetWikipediaSummary.Input,
            output_schema=GetWikipediaSummary.Output,
            test_input={"topic": "Artificial Intelligence"},
            test_output={"summary": "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals."},
        )

    def run(self, input_data: Input) -> BlockOutput:
        try:
            response = requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{input_data.topic}")
            response.raise_for_status()
            summary_data = response.json()
            
            yield "summary", summary_data['extract']

        except requests.exceptions.HTTPError as http_err:
            raise ValueError(f"HTTP error occurred: {http_err}")
        except requests.RequestException as e:
            raise ValueError(f"Request to Wikipedia API failed: {e}")
        except KeyError as e:
            raise ValueError(f"Error processing Wikipedia data: {e}")