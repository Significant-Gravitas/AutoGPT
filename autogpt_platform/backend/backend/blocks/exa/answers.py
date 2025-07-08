from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchema,
    CredentialsMetaInput,
    Requests,
    SchemaField,
)

from ._config import exa


class ExaAnswerBlock(Block):
    class Input(BlockSchema):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        query: str = SchemaField(
            description="The question or query to answer",
            placeholder="What is the latest valuation of SpaceX?",
        )
        text: bool = SchemaField(
            default=False,
            description="If true, the response includes full text content in the search results",
            advanced=True,
        )
        model: str = SchemaField(
            default="exa",
            description="The search model to use (exa or exa-pro)",
            placeholder="exa",
            advanced=True,
        )

    class Output(BlockSchema):
        answer: str = SchemaField(
            description="The generated answer based on search results"
        )
        citations: list[dict] = SchemaField(
            description="Search results used to generate the answer",
            default_factory=list,
        )
        cost_dollars: dict = SchemaField(
            description="Cost breakdown of the request", default_factory=dict
        )
        error: str = SchemaField(
            description="Error message if the request failed", default=""
        )

    def __init__(self):
        super().__init__(
            id="f8e7d6c5-b4a3-5c2d-9e1f-3a7b8c9d4e6f",
            description="Get an LLM answer to a question informed by Exa search results",
            categories={BlockCategory.SEARCH, BlockCategory.AI},
            input_schema=ExaAnswerBlock.Input,
            output_schema=ExaAnswerBlock.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        url = "https://api.exa.ai/answer"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": credentials.api_key.get_secret_value(),
        }

        # Build the payload
        payload = {
            "query": input_data.query,
            "text": input_data.text,
            "model": input_data.model,
        }

        try:
            # Note: This endpoint doesn't support streaming in our block implementation
            # If stream=True is requested, we still make a regular request
            response = await Requests().post(url, headers=headers, json=payload)
            data = response.json()

            yield "answer", data.get("answer", "")
            yield "citations", data.get("citations", [])
            yield "cost_dollars", data.get("costDollars", {})

        except Exception as e:
            yield "error", str(e)
            yield "answer", ""
            yield "citations", []
            yield "cost_dollars", {}
