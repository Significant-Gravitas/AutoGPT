"""Firecrawl Alexandria: a catalogue of structured-data providers.

Alexandria puts thousands of data tools behind the Firecrawl API key: people
and company search and enrichment (Apollo, People Data Labs, FullEnrich, Data
Legion), jobs, finance and filings, public records, places, retail and more.
Discovery is free; running a tool costs the Firecrawl credits listed on it.

Both calls go to ``POST /v2/scrape`` with an ``alexandria`` body: discovery is
the built-in ``firecrawl/find-tools`` capability, execution is any provider's
capability. Docs: https://docs.firecrawl.dev/features/alexandria
"""

import uuid
from typing import Any

from pydantic import SecretStr

from backend.data.model import NodeExecutionStats
from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockCost,
    BlockCostType,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    Requests,
    SchemaField,
    cost,
)
from backend.util.exceptions import BlockInputError

from ._config import firecrawl

ALEXANDRIA_URL = "https://api.firecrawl.dev/v2/scrape"
# Firecrawl caps one Alexandria execution at 120 seconds.
MAX_TIMEOUT_SECONDS = 120
# Firecrawl bills in its own credits (1 credit ≈ $0.001), as in _config.py.
USD_PER_FIRECRAWL_CREDIT = 0.001

TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="firecrawl",
    api_key=SecretStr("mock-key"),
    title="Mock Firecrawl API key",
    expires_at=None,
)
TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}

_TEST_TOOL = {
    "provider": "apollo",
    "capability": "people/match",
    "description": "Resolve a person to a verified work email",
    "creditsCost": 5,
}


class AlexandriaError(Exception):
    """A refused Alexandria request, with Firecrawl's error code and message."""


async def call_alexandria(
    api_key: str,
    provider: str,
    capability: str,
    options: dict[str, Any] | None = None,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    """Run one Alexandria capability and return its result entry.

    The entry carries ``data``, ``records`` and ``creditsCost``, or an
    ``error`` when the capability failed. Raises AlexandriaError when
    Firecrawl refused the request.
    """
    call: dict[str, Any] = {"provider": provider, "capability": capability}
    if options:
        call["options"] = options
    body: dict[str, Any] = {"alexandria": [call]}
    if timeout_seconds:
        body["timeout"] = min(timeout_seconds, MAX_TIMEOUT_SECONDS) * 1000

    # One request id across retries, so Firecrawl runs (and bills) a retried
    # paid call once.
    response = await Requests(raise_for_status=False).post(
        ALEXANDRIA_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "x-request-id": str(uuid.uuid4()),
        },
        json=body,
    )
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    if not response.ok or not payload.get("success"):
        raise AlexandriaError(_describe_refusal(response.status, payload))

    results = payload.get("data", {}).get("alexandria") or []
    if not results:
        raise AlexandriaError("Firecrawl returned no Alexandria result")
    return results[0]


def _describe_refusal(status: int, payload: dict[str, Any]) -> str:
    message = payload.get("error") or f"HTTP {status}"
    code = payload.get("code")
    if code == "THIRD_PARTY_DATA_TERMS_REQUIRED":
        url = (payload.get("requiresAction") or {}).get("url")
        return (
            f"{message}. A Firecrawl organization admin must accept this "
            f"provider's terms first" + (f": {url}" if url else ".")
        )
    return f"{code}: {message}" if code else str(message)


def _charge(block: Block, result: dict[str, Any]) -> float:
    """Record what the call cost, then raise if the capability failed.

    A failed capability can still have been charged, so the cost is recorded
    first.
    """
    credits = float(result.get("creditsCost") or 0)
    block.merge_stats(
        NodeExecutionStats(
            provider_cost=credits * USD_PER_FIRECRAWL_CREDIT,
            provider_cost_type="cost_usd",
        )
    )
    if error := result.get("error"):
        raise AlexandriaError(
            f"{error.get('code', 'error')}: {error.get('message', 'unknown error')}"
        )
    return credits


# Discovery is free, so it must not inherit Firecrawl's paid provider cost: a
# paid entry would stop the block from running on an empty wallet.
@cost(BlockCost(cost_type=BlockCostType.RUN, cost_amount=0))
class FirecrawlFindDataToolsBlock(Block):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = firecrawl.credentials_field()
        query: str = SchemaField(
            description=(
                "What data you need, in plain words, e.g. 'work email for a "
                "person at a company' or 'sales leaders at Series B software "
                "companies'. Leave empty to browse by provider or category."
            ),
            default="",
        )
        providers: list[str] = SchemaField(
            description=(
                "Limit to these provider IDs, e.g. 'apollo' or "
                "'peopledatalabs'. Alone, lists that provider's tools."
            ),
            default_factory=list,
        )
        capabilities: list[str] = SchemaField(
            description=(
                "Capability IDs to inspect within the providers above, e.g. "
                "'people/search'. Returns each one's full contract."
            ),
            default_factory=list,
            advanced=True,
        )
        categories: list[str] = SchemaField(
            description=(
                "Category IDs, e.g. 'people' or 'company'. Lists the "
                "providers in those categories."
            ),
            default_factory=list,
            advanced=True,
        )
        include_contracts: bool = SchemaField(
            description=(
                "Include each tool's inputs and response shape, so its "
                "options can be filled in for 'Firecrawl Run Data Tool'."
            ),
            default=True,
            advanced=True,
        )
        limit: int = SchemaField(
            description="Most results to return (1-100)",
            default=10,
            ge=1,
            le=100,
            advanced=True,
        )
        offset: int = SchemaField(
            description="Results to skip, for paging",
            default=0,
            ge=0,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        results: list[dict[str, Any]] = SchemaField(
            description=(
                "Matching tools, each with its provider, capability, "
                "description, price in Firecrawl credits and, when asked "
                "for, its input options. When browsing, providers or "
                "categories instead."
            )
        )
        result: dict[str, Any] = SchemaField(description="Each result in turn")
        level: str = SchemaField(
            description="What the results are: tools, providers, groups or categories"
        )
        total: int = SchemaField(description="Total number of matches")

    def __init__(self):
        super().__init__(
            id="3b1d6b0e-4f0a-4b6e-9a51-6d2f8c7e0a11",
            description=(
                "Finds structured-data tools in Firecrawl Alexandria, free of "
                "charge: people and company search and enrichment (work "
                "emails, job titles, firmographics via Apollo, People Data "
                "Labs, FullEnrich), jobs, finance and filings, public records, "
                "places, retail and thousands more. Pass a tool's provider, "
                "capability and options to 'Firecrawl Run Data Tool'."
            ),
            categories={BlockCategory.SEARCH, BlockCategory.DATA},
            input_schema=self.Input,
            output_schema=self.Output,
            effect=BlockEffect.READ,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "query": "work email for a person at a company",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("results", [_TEST_TOOL]),
                ("result", _TEST_TOOL),
                ("level", "tools"),
                ("total", 1),
            ],
            test_mock={
                "call_alexandria": lambda *args, **kwargs: {
                    "provider": "firecrawl",
                    "capability": "find-tools",
                    "creditsCost": 0,
                    "data": {"level": "tools", "items": [_TEST_TOOL], "total": 1},
                }
            },
        )

    @staticmethod
    async def call_alexandria(*args, **kwargs) -> dict[str, Any]:
        return await call_alexandria(*args, **kwargs)

    @staticmethod
    def build_options(input_data: "FirecrawlFindDataToolsBlock.Input") -> dict:
        query = input_data.query.strip()
        if query or input_data.providers or input_data.capabilities:
            level = "tools"
        elif input_data.categories:
            level = "providers"
        else:
            level = "categories"

        options: dict[str, Any] = {"level": level, "limit": input_data.limit}
        if input_data.offset:
            options["offset"] = input_data.offset
        if level != "categories":
            if query:
                options["query"] = query
            for key in ("providers", "capabilities", "categories"):
                if values := getattr(input_data, key):
                    options[key] = values
        if level == "tools":
            options["expand"] = (
                ["options", "response"] if input_data.include_contracts else []
            )
        return options

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        result = await self.call_alexandria(
            credentials.api_key.get_secret_value(),
            "firecrawl",
            "find-tools",
            self.build_options(input_data),
        )
        _charge(self, result)
        page = result.get("data") or {}
        items = page.get("items") or []
        yield "results", items
        for item in items:
            yield "result", item
        yield "level", page.get("level", "")
        yield "total", page.get("total", len(items))


class FirecrawlRunDataToolBlock(Block):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = firecrawl.credentials_field()
        provider: str = SchemaField(
            description="Provider ID from 'Firecrawl Find Data Tools', e.g. 'apollo'"
        )
        capability: str = SchemaField(
            description=(
                "Capability ID from 'Firecrawl Find Data Tools', e.g. "
                "'people/search'"
            )
        )
        options: dict[str, Any] = SchemaField(
            description=(
                "The tool's inputs, exactly as its contract declares them "
                "(see the tool's options in 'Firecrawl Find Data Tools')"
            ),
            default_factory=dict,
        )
        timeout: int = SchemaField(
            description="Seconds to wait for the provider (at most 120)",
            default=60,
            ge=1,
            le=MAX_TIMEOUT_SECONDS,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        data: Any = SchemaField(description="The records the tool returned")
        records: int = SchemaField(description="How many records were returned")
        credits_cost: float = SchemaField(
            description="Firecrawl credits this call cost"
        )

    def __init__(self):
        super().__init__(
            id="9c4f2a7e-1d3b-4e8a-b6c5-2f7a0e9d1c34",
            description=(
                "Runs a Firecrawl Alexandria data tool and returns its typed, "
                "sourced records: find people by title and company, get "
                "verified work emails, enrich a company from its domain, and "
                "thousands more. Find the tool and its inputs with 'Firecrawl "
                "Find Data Tools' first. Costs the Firecrawl credits listed on "
                "the tool."
            ),
            categories={BlockCategory.SEARCH, BlockCategory.DATA},
            input_schema=self.Input,
            output_schema=self.Output,
            effect=BlockEffect.READ,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "provider": "apollo",
                "capability": "people/search",
                "options": {"titles": ["VP of Sales"], "limit": 1},
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("data", [{"name": "Amara Okafor", "title": "VP of Sales"}]),
                ("records", 1),
                ("credits_cost", 5.0),
            ],
            test_mock={
                "call_alexandria": lambda *args, **kwargs: {
                    "provider": "apollo",
                    "capability": "people/search",
                    "creditsCost": 5,
                    "records": 1,
                    "data": [{"name": "Amara Okafor", "title": "VP of Sales"}],
                }
            },
        )

    @staticmethod
    async def call_alexandria(*args, **kwargs) -> dict[str, Any]:
        return await call_alexandria(*args, **kwargs)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        provider = input_data.provider.strip()
        capability = input_data.capability.strip()
        if not provider or not capability:
            raise BlockInputError(
                message="Provider and capability are both required",
                block_name=self.name,
                block_id=self.id,
            )
        result = await self.call_alexandria(
            credentials.api_key.get_secret_value(),
            provider,
            capability,
            input_data.options,
            input_data.timeout,
        )
        credits = _charge(self, result)
        data = result.get("data")
        records = result.get("records")
        if records is None:
            records = len(data) if isinstance(data, list) else int(data is not None)
        yield "data", data
        yield "records", records
        yield "credits_cost", credits
