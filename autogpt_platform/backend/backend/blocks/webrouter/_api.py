from typing import Any

from web_router import AsyncWebRouter, Provider

from backend.sdk import BaseModel, SchemaField

# WebRouter's own default chain puts first any backend whose API key is set in
# the environment. This server keeps its own Tavily, Exa, Jina and Parallel keys
# there for the blocks of those providers, so the WebRouter blocks give it a
# fixed chain of keyless tiers instead and never spend those keys. Exa's and
# Parallel's entries are their keyless MCP servers, which read no key.
SEARCH_CHAIN: list[Any] = [
    (Provider.NIMBLE.value, {"allow_keyless": True}),
    Provider.YOU_MCP.value,
    Provider.EXA_MCP.value,
    Provider.PARALLEL_MCP.value,
    (Provider.KEENABLE.value, {"allow_keyless": True}),
]
EXTRACT_CHAIN: list[Any] = [
    (Provider.NIMBLE.value, {"allow_keyless": True}),
    (Provider.FIRECRAWL.value, {"allow_keyless": True}),
    (Provider.KEENABLE.value, {"allow_keyless": True}),
]

# Seconds a whole call may take, fallbacks included.
SEARCH_TIMEOUT = 60.0
EXTRACT_TIMEOUT = 120.0


def keyless_router() -> AsyncWebRouter:
    """A router that tries only the keyless chains above. Build one per call."""
    return AsyncWebRouter(
        provider_config={
            "auto": {"chain": {"search": SEARCH_CHAIN, "extract": EXTRACT_CHAIN}}
        }
    )


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
