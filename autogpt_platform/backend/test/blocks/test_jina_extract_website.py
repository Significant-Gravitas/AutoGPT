from typing import cast

import pytest

from backend.blocks.jina._auth import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    JinaCredentialsInput,
)
from backend.blocks.jina.search import ExtractWebsiteContentBlock, SearchTheWebBlock
from backend.util.exceptions import BlockExecutionError
from backend.util.request import HTTPClientError


@pytest.mark.asyncio
async def test_extract_website_content_returns_content(monkeypatch):
    block = ExtractWebsiteContentBlock()
    input_data = block.Input(
        url="https://example.com",
        credentials=cast(JinaCredentialsInput, TEST_CREDENTIALS_INPUT),
        raw_content=True,
    )

    async def fake_get_request(url, json=False, headers=None):
        assert url == "https://example.com"
        assert headers == {}
        return "page content"

    monkeypatch.setattr(block, "get_request", fake_get_request)

    results = [
        output
        async for output in block.run(
            input_data=input_data, credentials=TEST_CREDENTIALS
        )
    ]

    assert ("content", "page content") in results
    assert all(key != "error" for key, _ in results)


@pytest.mark.asyncio
async def test_extract_website_content_handles_http_error(monkeypatch):
    block = ExtractWebsiteContentBlock()
    input_data = block.Input(
        url="https://example.com",
        credentials=cast(JinaCredentialsInput, TEST_CREDENTIALS_INPUT),
        raw_content=False,
    )

    async def fake_get_request(_url, json=False, headers=None):
        raise HTTPClientError("HTTP 400 Error: Bad Request", 400)

    monkeypatch.setattr(block, "get_request", fake_get_request)

    results = [
        output
        async for output in block.run(
            input_data=input_data, credentials=TEST_CREDENTIALS
        )
    ]

    assert ("content", "page content") not in results
    error_messages = [value for key, value in results if key == "error"]
    assert error_messages
    assert "Client error (400)" in error_messages[0]
    assert "https://example.com" in error_messages[0]


@pytest.mark.asyncio
async def test_search_the_web_returns_empty_results_for_no_matches(monkeypatch):
    block = SearchTheWebBlock()
    input_data = block.Input(
        query="PES 2013 player stats historical",
        credentials=cast(JinaCredentialsInput, TEST_CREDENTIALS_INPUT),
    )

    async def fake_get_request(_url, headers=None, json=False):
        raise HTTPClientError(
            'HTTP 422 Error: Unprocessable Entity, Body: {"message":"No search results available for query"}',
            422,
        )

    monkeypatch.setattr(block, "get_request", fake_get_request)

    results = [
        output
        async for output in block.run(
            input_data=input_data, credentials=TEST_CREDENTIALS
        )
    ]

    assert results == [("results", "")]


@pytest.mark.asyncio
async def test_search_the_web_still_raises_for_other_client_errors(monkeypatch):
    block = SearchTheWebBlock()
    input_data = block.Input(
        query="Artificial Intelligence",
        credentials=cast(JinaCredentialsInput, TEST_CREDENTIALS_INPUT),
    )

    async def fake_get_request(_url, headers=None, json=False):
        raise HTTPClientError("HTTP 401 Error: Unauthorized", 401)

    monkeypatch.setattr(block, "get_request", fake_get_request)

    with pytest.raises(BlockExecutionError, match="Search failed: HTTP 401 Error"):
        [
            output
            async for output in block.run(
                input_data=input_data, credentials=TEST_CREDENTIALS
            )
        ]
