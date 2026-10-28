"""Treg block that calls any catalog tool, or one of the team's own tools."""

from enum import Enum
from typing import Any

from backend.sdk import (
    APIKeyCredentials,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaOutput,
    SchemaField,
)

from ._api import HttpMethod, TregError, validate_tool
from ._base import MAX_COST_CEILING_USD, TregBlockBase, TregInput
from ._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from ._test import TEST_CALL_ID, mock_response


class TregCallMethod(str, Enum):
    AUTO = "AUTO"
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"


TEST_ANSWER = {"data": {"username": "tiktok", "followers": 92000000}}


class TregCallToolBlock(TregBlockBase):
    class Input(TregInput):
        tool: str = SchemaField(
            description=(
                "What to call: a catalog tool id, e.g. 'tikhub.tiktok.user.profile', "
                "as Treg Search Catalog lists it; or a tool your Treg team "
                "registered, with its path, e.g. 'intercom/conversations'."
            ),
            min_length=1,
        )
        query: dict[str, Any] = SchemaField(
            default_factory=dict,
            description='Query parameters, e.g. {"uniqueId": "tiktok"}. A list '
            "value repeats the parameter.",
        )
        body: Any = SchemaField(
            default=None,
            description="A JSON body, for tools called with POST. Treg Get Tool "
            "lists the fields a tool takes.",
        )
        method: TregCallMethod = SchemaField(
            default=TregCallMethod.AUTO,
            advanced=True,
            description="The HTTP method. AUTO sends POST when there is a body, "
            "GET otherwise.",
        )
        headers: dict[str, str] = SchemaField(
            default_factory=dict,
            advanced=True,
            description="Extra headers, e.g. a provider option or Treg's "
            "'X-Treg-Route-Verify: true'. Your Treg key is added for you.",
        )
        max_cost_usd: float = SchemaField(
            default=1.0,
            ge=0,
            le=MAX_COST_CEILING_USD,
            advanced=True,
            description=(
                "The most this call may cost, in USD. Treg refuses a call priced "
                "above it, without charge."
            ),
        )

    class Output(BlockSchemaOutput):
        response: Any = SchemaField(
            description="The provider's answer, exactly as it sent it."
        )
        status: int = SchemaField(description="The provider's HTTP status.")
        cost_usd: float = SchemaField(
            description="What Treg charged, in USD. For a task still running "
            "(see async_task) this is the amount reserved, settled when it ends."
        )
        call_id: str = SchemaField(description="Treg's id for this call.")
        served_by: str = SchemaField(
            description="The tool that answered, for a routed tool. Emitted only "
            "when Treg names one."
        )
        cached: bool = SchemaField(
            description="True when Treg answered from its archive of the same "
            "question, at a tenth of the price."
        )
        async_task: dict[str, Any] = SchemaField(
            description=(
                "For a tool that starts a background task (a video or image "
                "generation): how to check on it with another call. Emitted only "
                "then."
            )
        )

    def __init__(self):
        super().__init__(
            id="c1e7a3f9-5b2d-4c68-8e0a-6d4f2b9c7e15",
            description=(
                "Call any of Treg's 4,000+ tools by id (SEO, social, ads, people "
                "and company data, scraping, generation), or a tool your Treg team "
                "registered. Find tool ids and prices with Treg Search Catalog."
            ),
            categories={BlockCategory.SEARCH, BlockCategory.DEVELOPER_TOOLS},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "tool": "tikhub.tiktok.user.profile",
                "query": {"uniqueId": "tiktok"},
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("response", TEST_ANSWER),
                ("status", 200),
                ("cost_usd", 0.001),
                ("call_id", TEST_CALL_ID),
                ("cached", False),
            ],
            test_mock={
                "call_treg": lambda *args, **kwargs: mock_response(
                    TEST_ANSWER, cost_usd=0.001
                )
            },
            # A tool can be anything, including a write through a team's own
            # registered API, so the block is treated as an outside action.
            effect=BlockEffect.EXTERNAL,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        tool = validate_tool(input_data.tool)
        method = call_method(input_data.method, input_data.body)
        if method == "GET" and input_data.body is not None:
            raise TregError(
                "A GET call cannot carry a body; use POST or drop the body."
            )
        response = await self.request(
            credentials,
            method,
            f"call/{tool}",
            params=input_data.query,
            body=input_data.body,
            headers={
                **input_data.headers,
                "X-Treg-Route-Max-Cost": f"{input_data.max_cost_usd:g}",
            },
        )
        if response.upstream_failed:
            raise TregError(
                f"{tool} reported a failure inside a successful answer "
                f"(call id {response.call_id}), so there is no result. Any charge "
                "is what the provider billed for the attempt."
            )
        yield "response", response.body
        yield "status", response.status
        yield "cost_usd", response.cost_usd
        if response.call_id:
            yield "call_id", response.call_id
        if response.served_by:
            yield "served_by", response.served_by
        yield "cached", response.cached
        if response.async_task:
            yield "async_task", response.async_task


def call_method(method: TregCallMethod, body: Any) -> HttpMethod:
    if method == TregCallMethod.GET:
        return "GET"
    if method == TregCallMethod.POST:
        return "POST"
    if method == TregCallMethod.PUT:
        return "PUT"
    if method == TregCallMethod.PATCH:
        return "PATCH"
    if method == TregCallMethod.DELETE:
        return "DELETE"
    return "GET" if body is None else "POST"
