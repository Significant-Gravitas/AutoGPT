"""
Exa Code Context Block

Provides code search capabilities to find relevant code snippets and examples
from open source repositories, documentation, and Stack Overflow.
"""

from typing import Union

from pydantic import BaseModel

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    Requests,
    SchemaField,
)

from ._config import exa


class CodeContextResponse(BaseModel):
    """Stable output model for code context responses."""

    request_id: str
    query: str
    response: str
    results_count: int
    cost_dollars: str
    search_time: float
    output_tokens: int

    @classmethod
    def from_api(cls, data: dict) -> "CodeContextResponse":
        """Convert API response to our stable model."""
        return cls(
            request_id=data.get("requestId", ""),
            query=data.get("query", ""),
            response=data.get("response", ""),
            results_count=data.get("resultsCount", 0),
            cost_dollars=data.get("costDollars", ""),
            search_time=data.get("searchTime", 0.0),
            output_tokens=data.get("outputTokens", 0),
        )


class ExaCodeContextBlock(Block):
    """Get relevant code snippets and examples from open source repositories."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        query: str = SchemaField(
            description="Search query to find relevant code snippets. Describe what you're trying to do or what code you're looking for.",
            placeholder="how to use React hooks for state management",
        )
        tokens_num: Union[str, int] = SchemaField(
            default="dynamic",
            description="Token limit for response. Use 'dynamic' for automatic sizing, 5000 for standard queries, or 10000 for comprehensive examples.",
            placeholder="dynamic",
        )

    class Output(BlockSchemaOutput):
        request_id: str = SchemaField(description="Unique identifier for this request")
        query: str = SchemaField(description="The search query used")
        response: str = SchemaField(
            description="Formatted code snippets and contextual examples with sources"
        )
        results_count: int = SchemaField(
            description="Number of code sources found and included"
        )
        cost_dollars: str = SchemaField(description="Cost of this request in dollars")
        search_time: float = SchemaField(
            description="Time taken to search in milliseconds"
        )
        output_tokens: int = SchemaField(description="Number of tokens in the response")

    def __init__(self):
        super().__init__(
            id="8f9e0d1c-2b3a-4567-8901-23456789abcd",
            description="Search billions of GitHub repos, docs, and Stack Overflow for relevant code examples",
            categories={BlockCategory.SEARCH, BlockCategory.DEVELOPER_TOOLS},
            input_schema=ExaCodeContextBlock.Input,
            output_schema=ExaCodeContextBlock.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        url = "https://api.exa.ai/context"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": credentials.api_key.get_secret_value(),
        }

        payload = {
            "query": input_data.query,
            "tokensNum": input_data.tokens_num,
        }

        response = await Requests().post(url, headers=headers, json=payload)
        data = response.json()

        context = CodeContextResponse.from_api(data)

        yield "request_id", context.request_id
        yield "query", context.query
        yield "response", context.response
        yield "results_count", context.results_count
        yield "cost_dollars", context.cost_dollars
        yield "search_time", context.search_time
        yield "output_tokens", context.output_tokens
