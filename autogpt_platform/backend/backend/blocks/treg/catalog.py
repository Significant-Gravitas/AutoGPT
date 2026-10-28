"""Treg blocks that search the tool catalog and describe a tool."""

import re
from typing import Any

from pydantic import BaseModel, Field

from backend.sdk import (
    APIKeyCredentials,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaOutput,
    SchemaField,
)

from ._api import TregError
from ._base import TregBlockBase, TregInput
from ._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from ._test import mock_response

_TOOL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class TregTool(BaseModel):
    """A catalog tool, with what it costs and how reliable it has been."""

    id: str = Field(description="The id to call it by, e.g. 'treg.people.search'")
    name: str = Field(default="", description="What it does, in a line")
    summary: str = Field(default="", description="A longer description")
    provider: str = Field(default="", description="Who serves it")
    kind: str = Field(
        default="",
        description="'routed' tools pick a provider for you; the rest are one "
        "provider's endpoint",
    )
    method: str = Field(default="GET", description="The HTTP method to call it with")
    price_usd: float | None = Field(
        default=None,
        description="Listed price per unit in USD; None when it varies by provider",
    )
    price_unit: str = Field(default="", description="What the price is per")
    price_note: str = Field(default="", description="How the price works")
    success_rate: float | None = Field(
        default=None, description="Share of calls Treg saw succeed, from 0 to 1"
    )
    call_example: str = Field(default="", description="A ready-made call line")

    @classmethod
    def from_catalog(cls, row: dict[str, Any]) -> "TregTool":
        cost = _dict(row.get("cost"))
        observed = _dict(row.get("observed"))
        varies = bool(cost.get("varies")) or row.get("kind") == "routed"
        return cls(
            id=str(row.get("id", "")),
            name=str(row.get("name") or ""),
            summary=str(row.get("summary") or ""),
            provider=str(row.get("provider_display") or row.get("provider") or ""),
            kind=str(row.get("kind") or ""),
            method=str(row.get("method") or "GET"),
            price_usd=None if varies else _number(cost.get("usd")),
            price_unit=str(cost.get("unit") or ""),
            price_note=str(cost.get("note") or ""),
            success_rate=_number(observed.get("ok_rate")),
            call_example=str(row.get("call_template") or ""),
        )


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


TEST_TOOL_ROW = {
    "id": "treg.people.email.find",
    "provider": "treg",
    "provider_display": "treg",
    "name": "people.email.find — routed: best of 29 providers, own keys first",
    "summary": "Find someone's work email",
    "kind": "routed",
    "method": "POST",
    "cost": {"type": "per_success", "usd": 0.0, "unit": "call", "varies": True},
    "observed": {"ok_rate": 0.9465},
    "call_template": "treg call treg.people.email.find --method POST --data '{}'",
    "input": {"bodyType": "json", "body": {"domain": {"type": "str"}}},
}
TEST_TOOL = TregTool.from_catalog(TEST_TOOL_ROW)


class TregSearchCatalogBlock(TregBlockBase):
    class Input(TregInput):
        query: str = SchemaField(
            description="The job you need done, e.g. 'backlinks for a domain' or "
            "'tiktok comments'. Search for the job, not a vendor.",
            min_length=1,
        )
        limit: int = SchemaField(
            default=10, ge=1, le=100, description="How many tools to return."
        )
        added_within_days: int | None = SchemaField(
            default=None,
            ge=1,
            advanced=True,
            description="Only tools added in the last this-many days.",
        )

    class Output(BlockSchemaOutput):
        tools: list[TregTool] = SchemaField(
            description="Matching tools, best first, with prices and success rates."
        )
        count: int = SchemaField(description="How many tools came back.")
        total: int = SchemaField(description="How many tools matched in all.")

    def __init__(self):
        super().__init__(
            id="e8a2c6f0-9b4d-4e31-a7c5-2f6d8b0e4a19",
            description=(
                "Search Treg's catalog of 4,000+ tools by the job you need done, with "
                "each tool's price and success rate. Searching is free; call a tool "
                "you find with Treg Call Tool."
            ),
            categories={BlockCategory.SEARCH, BlockCategory.DEVELOPER_TOOLS},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={"credentials": TEST_CREDENTIALS_INPUT, "query": "work email"},
            test_credentials=TEST_CREDENTIALS,
            test_output=[("tools", [TEST_TOOL]), ("count", 1), ("total", 123)],
            test_mock={
                "call_treg": lambda *args, **kwargs: mock_response(
                    {"query": "work email", "total": 123, "results": [TEST_TOOL_ROW]}
                )
            },
            effect=BlockEffect.READ,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        response = await self.request(
            credentials,
            "GET",
            "catalog/search",
            params={
                "q": input_data.query,
                "limit": input_data.limit,
                "added_within_days": input_data.added_within_days,
            },
        )
        body = _dict(response.body)
        listed = _list(body.get("results"))
        tools = [TregTool.from_catalog(row) for row in listed if isinstance(row, dict)]
        yield "tools", tools
        yield "count", len(tools)
        total = body.get("total")
        yield "total", total if isinstance(total, int) else len(tools)


class TregProviderOption(BaseModel):
    """Another provider that does the same job, and how it compares."""

    id: str = Field(description="The tool id to call this provider directly")
    provider: str = Field(default="", description="The provider")
    price_usd: float | None = Field(default=None, description="Price per unit in USD")
    success_rate: float | None = Field(
        default=None, description="Share of calls Treg saw succeed, from 0 to 1"
    )
    hit_rate: float | None = Field(
        default=None, description="Share of successful calls that found an answer"
    )


class TregGetToolBlock(TregBlockBase):
    class Input(TregInput):
        tool_id: str = SchemaField(
            description="The tool's id, e.g. 'treg.people.email.find', as Treg "
            "Search Catalog lists it.",
            min_length=1,
        )

    class Output(BlockSchemaOutput):
        tool: TregTool = SchemaField(description="The tool, its price and reliability.")
        inputs: dict[str, Any] = SchemaField(
            description="The query parameters and body fields it takes, with notes."
        )
        providers: list[TregProviderOption] = SchemaField(
            description="Every provider of the same job, to compare price and "
            "reliability before you choose."
        )

    def __init__(self):
        super().__init__(
            id="7b3d9f1e-0c5a-4e82-b6d4-8a1f3c9e5b27",
            description=(
                "Look up one Treg tool before calling it: the inputs it takes, its "
                "price, and every other provider of the same job. Free."
            ),
            categories={BlockCategory.SEARCH, BlockCategory.DEVELOPER_TOOLS},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "tool_id": "treg.people.email.find",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("tool", TEST_TOOL),
                ("inputs", TEST_TOOL_ROW["input"]),
                ("providers", [TEST_PROVIDER_OPTION]),
            ],
            test_mock={
                "call_treg": lambda *args, **kwargs: mock_response(
                    {"endpoint": TEST_TOOL_ROW, "siblings": [TEST_SIBLING_ROW]}
                )
            },
            effect=BlockEffect.READ,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        tool_id = input_data.tool_id.strip()
        if not _TOOL_ID.match(tool_id):
            raise TregError(f"'{tool_id}' is not a Treg tool id.")
        response = await self.request(
            credentials, "GET", f"catalog/endpoints/{tool_id}"
        )
        body = _dict(response.body)
        endpoint = body.get("endpoint")
        if not isinstance(endpoint, dict):
            raise TregError(f"Treg returned no details for '{tool_id}'.")
        yield "tool", TregTool.from_catalog(endpoint)
        inputs = endpoint.get("input")
        yield "inputs", inputs if isinstance(inputs, dict) else {}
        siblings = _list(body.get("siblings"))
        yield "providers", [
            provider_option(row) for row in siblings if isinstance(row, dict)
        ]


def provider_option(row: dict[str, Any]) -> TregProviderOption:
    cost = _dict(row.get("cost"))
    observed = _dict(row.get("observed"))
    return TregProviderOption(
        id=str(row.get("id", "")),
        provider=str(row.get("provider_display") or row.get("provider") or ""),
        price_usd=_number(cost.get("usd")),
        success_rate=_number(observed.get("ok_rate")),
        hit_rate=_number(observed.get("hit_rate")),
    )


TEST_SIBLING_ROW = {
    "id": "hunter.email.finder",
    "provider": "hunter",
    "provider_display": "Hunter",
    "cost": {"type": "per_success", "usd": 0.0245},
    "observed": {"ok_rate": 0.99, "hit_rate": 0.67},
}
TEST_PROVIDER_OPTION = provider_option(TEST_SIBLING_ROW)
