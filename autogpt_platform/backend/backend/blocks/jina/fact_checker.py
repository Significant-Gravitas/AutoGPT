from urllib.parse import quote

from backend.blocks.jina._auth import (
    JinaCredentials,
    JinaCredentialsField,
    JinaCredentialsInput,
)
from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField
from backend.util.request import Requests


class FactCheckerBlock(Block):
    class Input(BlockSchema):
        statement: str = SchemaField(
            description="The statement to check for factuality"
        )
        credentials: JinaCredentialsInput = JinaCredentialsField()

    class Output(BlockSchema):
        factuality: float = SchemaField(
            description="The factuality score of the statement"
        )
        result: bool = SchemaField(description="The result of the factuality check")
        reason: str = SchemaField(description="The reason for the factuality result")
        error: str = SchemaField(description="Error message if the check fails")

    def __init__(self):
        super().__init__(
            id="d38b6c5e-9968-4271-8423-6cfe60d6e7e6",
            description="This block checks the factuality of a given statement using Jina AI's Grounding API. "
            "The block will retry automatically for transient failures.",
            categories={BlockCategory.SEARCH},
            input_schema=FactCheckerBlock.Input,
            output_schema=FactCheckerBlock.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: JinaCredentials, **kwargs
    ) -> BlockOutput:
        encoded_statement = quote(input_data.statement)
        url = f"https://g.jina.ai/{encoded_statement}"

        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {credentials.api_key.get_secret_value()}",
        }

        try:
            # The Requests class already has retry logic for 429, 500, 502, 503, 504, 408
            response = await Requests().get(url, headers=headers)

            try:
                data = response.json()
            except Exception as e:
                error_msg = f"Failed to parse JSON response: {str(e)}"
                yield "error", error_msg
                return

            if "data" in data:
                data = data["data"]
                # Ensure all required fields exist before accessing them
                try:
                    yield "factuality", data["factuality"]
                    yield "result", data["result"]
                    yield "reason", data["reason"]
                except KeyError as e:
                    error_msg = f"Missing expected field in response: {str(e)}. Response data: {data}"
                    yield "error", error_msg
            else:
                error_msg = f"Unexpected response format. Expected 'data' key not found in response: {data}"
                yield "error", error_msg
        except Exception as e:
            error_msg = f"Unexpected error during fact checking: {str(e)}"
            yield "error", error_msg
