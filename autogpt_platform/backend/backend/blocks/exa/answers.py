from backend.sdk import (
    APIKeyCredentials,
    BaseModel,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchema,
    CredentialsMetaInput,
    Requests,
    SchemaField,
)

from ._config import exa


class CostBreakdown(BaseModel):
    keywordSearch: float
    neuralSearch: float
    contentText: float
    contentHighlight: float
    contentSummary: float


class SearchBreakdown(BaseModel):
    search: float
    contents: float
    breakdown: CostBreakdown


class PerRequestPrices(BaseModel):
    neuralSearch_1_25_results: float
    neuralSearch_26_100_results: float
    neuralSearch_100_plus_results: float
    keywordSearch_1_100_results: float
    keywordSearch_100_plus_results: float


class PerPagePrices(BaseModel):
    contentText: float
    contentHighlight: float
    contentSummary: float


class CostDollars(BaseModel):
    total: float
    breakDown: list[SearchBreakdown]
    perRequestPrices: PerRequestPrices
    perPagePrices: PerPagePrices


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
        cost_dollars: CostDollars = SchemaField(
            description="Cost breakdown of the request"
        )
        error: str = SchemaField(
            description="Error message if the request failed", default=""
        )

    def __init__(self):
        super().__init__(
            id="b79ca4cc-9d5e-47d1-9d4f-e3a2d7f28df5",
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
