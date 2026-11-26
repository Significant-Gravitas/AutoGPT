"""Unit tests for the WebRouter Search block and the keyless router it uses.

The block's own test_input/test_mock case mocks the search away; these cover
the options it passes, how results are mapped, errors, and the chain.
"""

from typing import Any
from unittest.mock import AsyncMock

import pytest
from web_router import SearchResponse, SearchResult, WebRouterError

from backend.blocks.webrouter._api import keyless_router
from backend.blocks.webrouter.search import WebRouterSearchBlock
from backend.util.exceptions import BlockExecutionError

RESULTS = [
    SearchResult(url="https://a.example", title="A", description="snippet A"),
    SearchResult(url="https://b.example", title="", content="body B"),
    SearchResult(url="https://c.example", title="C", published_date="2026-11-01"),
]


async def collect(block: WebRouterSearchBlock, **inputs: Any) -> dict[str, Any]:
    outputs: dict[str, Any] = {"result": []}
    async for name, value in block.run(block.Input.model_validate(inputs)):
        if name == "result":
            outputs["result"].append(value)
        else:
            outputs[name] = value
    return outputs


def block_returning(results: list[SearchResult]) -> WebRouterSearchBlock:
    block = WebRouterSearchBlock()
    block._search = AsyncMock(
        return_value=SearchResponse(provider="you-mcp", query="q", results=results)
    )
    return block


@pytest.mark.asyncio
async def test_passes_the_query_and_options():
    block = block_returning(RESULTS)
    await collect(block, query="q", max_results=3, include_domains=["a.example"])
    block._search.assert_awaited_once_with("q", 3, ["a.example"])


@pytest.mark.asyncio
async def test_maps_results_and_names_the_provider():
    outputs = await collect(block_returning(RESULTS), query="q", max_results=3)

    assert [r.url for r in outputs["results"]] == [r.url for r in RESULTS]
    assert outputs["result"] == outputs["results"]
    assert [r.snippet for r in outputs["results"]] == ["snippet A", "body B", ""]
    assert outputs["results"][2].published_date == "2026-11-01"
    assert outputs["provider"] == "you-mcp"
    assert outputs["context"].startswith("[A](https://a.example)\nsnippet A")
    assert "[https://b.example](https://b.example)\nbody B" in outputs["context"]


@pytest.mark.asyncio
async def test_never_returns_more_than_max_results():
    outputs = await collect(block_returning(RESULTS), query="q", max_results=2)
    assert len(outputs["results"]) == 2


@pytest.mark.asyncio
async def test_no_results_yields_empty_outputs():
    outputs = await collect(block_returning([]), query="q")
    assert outputs["results"] == []
    assert outputs["result"] == []
    assert outputs["context"] == ""


@pytest.mark.asyncio
async def test_webrouter_error_becomes_a_block_error():
    block = WebRouterSearchBlock()
    block._search = AsyncMock(side_effect=WebRouterError("every provider failed"))
    with pytest.raises(BlockExecutionError, match="every provider failed"):
        await collect(block, query="q")


@pytest.mark.asyncio
async def test_router_ignores_provider_keys_in_the_environment(
    monkeypatch: pytest.MonkeyPatch,
):
    for var in ("TAVILY_API_KEY", "EXA_API_KEY", "JINA_API_KEY", "PARALLEL_API_KEY"):
        monkeypatch.setenv(var, "platform-key")

    async with keyless_router() as router:
        search = router.get_provider("auto").members
        extract = router.get_extract_provider("auto").members

    assert search == ("nimble", "you-mcp", "exa-mcp", "parallel-mcp", "keenable")
    assert extract == ("nimble", "firecrawl", "keenable")
