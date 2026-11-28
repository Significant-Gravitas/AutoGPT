import os
from typing import Any, Literal

from web_router import AsyncWebRouter, Provider, WebRouterError

from backend.sdk import BaseModel, SchemaField

# WebRouter's own default chain puts first any backend whose API key is set in
# the environment. This server keeps its own Tavily, Exa, Jina and Parallel keys
# there for the blocks of those providers, so the WebRouter blocks give it a
# fixed chain of keyless tiers instead and never spend those keys. Exa's and
# Parallel's entries are their keyless MCP servers, which read no key.
SEARCH_CHAIN: list[tuple[str, dict[str, Any]]] = [
    (Provider.NIMBLE.value, {"allow_keyless": True}),
    (Provider.YOU_MCP.value, {}),
    (Provider.EXA_MCP.value, {}),
    (Provider.PARALLEL_MCP.value, {}),
    (Provider.KEENABLE.value, {"allow_keyless": True}),
]
EXTRACT_CHAIN: list[tuple[str, dict[str, Any]]] = [
    (Provider.NIMBLE.value, {"allow_keyless": True}),
    (Provider.FIRECRAWL.value, {"allow_keyless": True}),
    (Provider.KEENABLE.value, {"allow_keyless": True}),
]

# A keyless tier still uses its provider's key when that key is set, so a
# member whose key is in the environment is left out of the chain.
KEY_VARIABLES = {
    Provider.NIMBLE.value: "NIMBLE_API_KEY",
    Provider.FIRECRAWL.value: "FIRECRAWL_API_KEY",
    Provider.KEENABLE.value: "KEENABLE_API_KEY",
}

# Seconds a whole call may take, fallbacks included.
SEARCH_TIMEOUT = 60.0
EXTRACT_TIMEOUT = 120.0

Method = Literal["search", "extract"]


def keyless_router(method: Method) -> AsyncWebRouter:
    """A router whose `method` tries only keyless tiers. Build one per call."""
    chain = keyless_chain(SEARCH_CHAIN if method == "search" else EXTRACT_CHAIN)
    if not chain:
        raise WebRouterError(
            f"No keyless {method} provider is left: every one has its API key "
            "set in this server's environment"
        )
    return AsyncWebRouter(provider_config={"auto": {"chain": {method: chain}}})


def keyless_chain(
    chain: list[tuple[str, dict[str, Any]]],
) -> list[tuple[str, dict[str, Any]]]:
    """The members of `chain` whose own API key is not set in the environment."""
    return [
        (name, options)
        for name, options in chain
        if not os.environ.get(KEY_VARIABLES.get(name, ""), "").strip()
    ]


class WebRouterSearchResult(BaseModel):
    title: str = SchemaField(description="Title of the result page")
    url: str = SchemaField(description="URL of the result page")
    snippet: str = SchemaField(description="Text from the page that matches the query")
    published_date: str | None = SchemaField(
        description="Publication date, when the search provider gave one",
        default=None,
    )


class WebRouterPage(BaseModel):
    url: str = SchemaField(description="URL of the page")
    title: str = SchemaField(description="Title of the page, when found")
    content: str = SchemaField(description="The page content as markdown")


class WebRouterFailure(BaseModel):
    url: str = SchemaField(description="URL that could not be read")
    error: str = SchemaField(description="Why it could not be read")
