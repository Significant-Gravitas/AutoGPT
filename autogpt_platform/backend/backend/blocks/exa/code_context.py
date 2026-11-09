"""
Exa Code Context Block

Calls Exa's ``/context`` endpoint for code snippets and examples from open
source repositories, documentation, and Stack Overflow. Exa no longer documents
``/context`` (it is absent from the API spec and its reference page redirects to
the "Code & Docs" search guide), so the block is kept for saved workflows and
points new users at Exa Search.
"""

from typing import Any

from pydantic import BaseModel

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

from ._config import exa
from ._exa_rest import EXA_API_BASE, exa_request
from ._test import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from .helpers import merge_exa_cost


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
            cost_dollars=cost_dollars_text(data.get("costDollars")),
            search_time=data.get("searchTime", 0.0),
            output_tokens=data.get("outputTokens", 0),
        )


def cost_dollars_text(raw: Any) -> str:
    """``costDollars`` as text: Exa sent a bare string here, but its other
    endpoints send ``{"total": ...}``, so accept both."""
    if raw is None:
        return ""
    if isinstance(raw, dict):
        total = raw.get("total")
        return "" if total is None else str(total)
    return str(raw)


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
        tokens_num: str | int = SchemaField(
            default="dynamic",
            description="Token limit for response. Use 'dynamic' for automatic sizing, 5000 for standard queries, or 10000 for comprehensive examples. (In Exa Search the equivalent is text max characters.)",
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
            description="Legacy: Exa no longer documents its code context API; for "
            "new workflows use the Exa Search block, which Exa's Code & Docs guide "
            "recommends (e.g. type 'lite' with highlights). Searches GitHub repos, "
            "docs and Stack Overflow for relevant code examples.",
            categories={BlockCategory.SEARCH, BlockCategory.DEVELOPER_TOOLS},
            input_schema=ExaCodeContextBlock.Input,
            output_schema=ExaCodeContextBlock.Output,
            effect=BlockEffect.READ,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "query": "how to use React hooks for state management",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("request_id", "req-1"),
                ("query", "how to use React hooks for state management"),
                ("response", "useState example"),
                ("results_count", 1),
                ("cost_dollars", "0.005"),
                ("search_time", 120.0),
                ("output_tokens", 42),
            ],
            test_mock={
                "_fetch_context": lambda *args, **kwargs: {
                    "requestId": "req-1",
                    "query": "how to use React hooks for state management",
                    "response": "useState example",
                    "resultsCount": 1,
                    "costDollars": "0.005",
                    "searchTime": 120.0,
                    "outputTokens": 42,
                }
            },
        )

    async def _fetch_context(
        self, credentials: APIKeyCredentials, payload: dict[str, Any]
    ) -> dict[str, Any]:
        return await exa_request(
            credentials, "POST", f"{EXA_API_BASE}/context", body=payload
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        payload = {
            "query": input_data.query,
            "tokensNum": input_data.tokens_num,
        }
        data = await self._fetch_context(credentials, payload)

        context = CodeContextResponse.from_api(data)

        yield "request_id", context.request_id
        yield "query", context.query
        yield "response", context.response
        yield "results_count", context.results_count
        yield "cost_dollars", context.cost_dollars
        yield "search_time", context.search_time
        yield "output_tokens", context.output_tokens

        # costDollars has been a bare numeric string like "0.005"; the helper
        # also accepts the {"total": ...} shape other endpoints use.
        merge_exa_cost(self, data)
