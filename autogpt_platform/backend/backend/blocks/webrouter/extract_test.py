"""Unit tests for the WebRouter Extract block: URL checks, outputs and errors.

The block's own test_input/test_mock case mocks the extraction away; these
cover what that mock skips.
"""

from typing import Any
from unittest.mock import AsyncMock

import pytest
from web_router import (
    ExtractedDocument,
    ExtractFailedError,
    ExtractFailure,
    ExtractResponse,
    WebRouterError,
)

from backend.blocks.webrouter.extract import WebRouterExtractBlock
from backend.util.exceptions import BlockExecutionError, BlockInputError

RESPONSE = ExtractResponse(
    provider="firecrawl",
    documents=[
        ExtractedDocument(url="https://a.example", title="A", content="# A"),
        ExtractedDocument(url="https://b.example", content="# B"),
    ],
    failures=[ExtractFailure(url="https://c.example", error="timed out")],
)


async def collect(block: WebRouterExtractBlock, **inputs: Any) -> dict[str, Any]:
    outputs: dict[str, Any] = {"result": []}
    async for name, value in block.run(block.Input.model_validate(inputs)):
        if name == "result":
            outputs["result"].append(value)
        else:
            outputs[name] = value
    return outputs


def block_returning(response: ExtractResponse) -> WebRouterExtractBlock:
    block = WebRouterExtractBlock()
    block._extract = AsyncMock(return_value=response)
    return block


@pytest.mark.asyncio
async def test_maps_pages_failures_and_provider():
    block = block_returning(RESPONSE)
    urls = ["https://a.example", "https://b.example", "https://c.example"]
    outputs = await collect(block, urls=urls)

    block._extract.assert_awaited_once_with(urls)
    assert [(p.url, p.title, p.content) for p in outputs["results"]] == [
        ("https://a.example", "A", "# A"),
        ("https://b.example", "", "# B"),
    ]
    assert outputs["result"] == outputs["results"]
    assert [(f.url, f.error) for f in outputs["failures"]] == [
        ("https://c.example", "timed out")
    ]
    assert outputs["provider"] == "firecrawl"


@pytest.mark.parametrize(
    "url", ["file:///etc/passwd", "ftp://a.example/x", "a.example", "javascript:x"]
)
@pytest.mark.asyncio
async def test_refuses_non_http_urls_before_any_call(url: str):
    block = block_returning(RESPONSE)
    with pytest.raises(BlockInputError, match="Only http and https"):
        await collect(block, urls=["https://a.example", url])
    block._extract.assert_not_awaited()


@pytest.mark.parametrize("count", [0, 11])
def test_takes_one_to_ten_urls(count: int):
    urls = [f"https://a.example/{i}" for i in range(count)]
    with pytest.raises(ValueError):
        WebRouterExtractBlock.Input.model_validate({"urls": urls})


@pytest.mark.asyncio
async def test_webrouter_error_becomes_a_block_error():
    block = WebRouterExtractBlock()
    block._extract = AsyncMock(side_effect=WebRouterError("every provider failed"))
    with pytest.raises(BlockExecutionError, match="every provider failed"):
        await collect(block, urls=["https://a.example"])


@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.asyncio
async def test_no_page_read_reports_every_url_as_a_failure(count: int):
    urls = [f"https://a.example/{i}" for i in range(count)]
    failures = [ExtractFailure(url=u, error="status 404") for u in urls]
    block = WebRouterExtractBlock()
    block._extract = AsyncMock(
        side_effect=ExtractFailedError(
            "firecrawl", "no URL could be extracted", failures
        )
    )

    outputs = await collect(block, urls=urls)

    assert outputs["results"] == []
    assert outputs["result"] == []
    assert [(f.url, f.error) for f in outputs["failures"]] == [
        (u, "status 404") for u in urls
    ]
    assert outputs["provider"] == "firecrawl"
