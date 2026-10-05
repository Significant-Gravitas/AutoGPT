"""Unit tests for the Search1API blocks.

Complements the ``test_mock`` harness in each block (exercised by
``test_available_blocks``) with direct coverage of request payloads, error
surfacing, per-query failure isolation in the batch block, and the
provider_cost reported for billing.
"""

import inspect
from unittest import mock

import pytest
from pydantic import SecretStr, ValidationError

from backend.blocks.search1api._api import (
    CREDIT_USD,
    SEARCH1API_API_URL,
    Search1APIClient,
    Search1APIResult,
    results_from_response,
    search_credits,
)
from backend.blocks.search1api.batch_search import Search1APIBatchSearchBlock
from backend.blocks.search1api.crawl import Search1APICrawlBlock
from backend.blocks.search1api.news import Search1APINewsBlock
from backend.blocks.search1api.search import Search1APISearchBlock
from backend.data.execution import ExecutionContext
from backend.sdk import APIKeyCredentials
from backend.util.exceptions import BlockExecutionError

TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="search1api",
    api_key=SecretStr("mock-search1api-api-key"),
    title="Mock Search1API API key",
    expires_at=None,
)
TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}


def _mock_block(block, mocks: dict):
    """Apply mocks to a block's methods, wrapping sync mocks as async."""
    for name, mock_fn in mocks.items():
        original = getattr(block, name)
        if inspect.iscoroutinefunction(original):

            async def async_mock(*args, _fn=mock_fn, **kwargs):
                return _fn(*args, **kwargs)

            setattr(block, name, async_mock)
        else:
            setattr(block, name, mock_fn)


def _raise(exc: Exception):
    """Return a callable which raises the given exception."""

    def _raiser(*args, **kwargs):
        raise exc

    return _raiser


def _capture_stats(block) -> list:
    merged: list = []
    block.merge_stats = lambda s: merged.append(s)  # type: ignore[assignment]
    return merged


async def _collect(block, input_data: dict) -> dict:
    """Run ``block.execute`` and gather its yielded outputs into a dict."""
    outputs: dict = {}
    async for name, value in block.execute(
        {"credentials": TEST_CREDENTIALS_INPUT, **input_data},
        credentials=TEST_CREDENTIALS,
        execution_context=ExecutionContext(),
    ):
        outputs.setdefault(name, []).append(value)
    return outputs


def _row(title="AutoGPT", link="https://agpt.co", content=None, **extra) -> dict:
    row = {"title": title, "link": link, "snippet": "AutoGPT builds agents."}
    if content is not None:
        row["content"] = content
    row.update(extra)
    return row


# ---------------------------------------------------------------------------
# Helpers in _api.py
# ---------------------------------------------------------------------------


def test_results_from_response_maps_link_to_url():
    results = results_from_response(
        {"results": [_row(content="page", published_date="2026-10-04")]}
    )
    assert results == [
        Search1APIResult(
            title="AutoGPT",
            url="https://agpt.co",
            snippet="AutoGPT builds agents.",
            content="page",
            published_date="2026-10-04",
        )
    ]


def test_results_from_response_rejects_missing_results():
    with pytest.raises(ValueError, match="missing results"):
        results_from_response({"searchParameters": {}})


def test_results_from_response_rejects_non_object_entries():
    with pytest.raises(ValueError, match="non-object result entry"):
        results_from_response({"results": [_row(), "not a result"]})


def test_search_credits_counts_crawled_pages_only_when_requested():
    results = results_from_response(
        {"results": [_row(content="a"), _row(content="b"), _row()]}
    )
    assert search_credits(results, crawl_results=0) == 1
    assert search_credits(results, crawl_results=3) == 3


@pytest.mark.asyncio
async def test_client_posts_with_bearer_auth():
    client = Search1APIClient(TEST_CREDENTIALS)
    resp = mock.Mock(ok=True)
    resp.json.return_value = {"results": []}
    client.requests.post = mock.AsyncMock(return_value=resp)

    out = await client.post("/search", {"query": "q"})

    assert out == {"results": []}
    args, kwargs = client.requests.post.call_args
    assert args[0] == f"{SEARCH1API_API_URL}/search"
    assert kwargs["json"] == {"query": "q"}
    assert client.requests.extra_headers == {
        "Authorization": "Bearer mock-search1api-api-key"
    }


@pytest.mark.asyncio
async def test_client_surfaces_api_error_message():
    client = Search1APIClient(TEST_CREDENTIALS)
    resp = mock.Mock(ok=False, status=401)
    resp.json.return_value = {
        "ok": False,
        "error": "Unauthorized: Invalid bearer credential",
        "message": "Unauthorized: Invalid bearer credential",
    }
    client.requests.post = mock.AsyncMock(return_value=resp)

    with pytest.raises(ValueError, match="HTTP 401: Unauthorized: Invalid bearer"):
        await client.post("/search", {"query": "q"})


@pytest.mark.asyncio
async def test_client_non_json_error_body():
    client = Search1APIClient(TEST_CREDENTIALS)
    resp = mock.Mock(ok=False, status=502)
    resp.json.side_effect = ValueError("not json")
    resp.text.return_value = "Bad Gateway"
    client.requests.post = mock.AsyncMock(return_value=resp)

    with pytest.raises(ValueError, match="HTTP 502: Bad Gateway"):
        await client.post("/crawl", {"url": "https://agpt.co"})


# ---------------------------------------------------------------------------
# Search and News blocks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "block_cls, method, endpoint, service",
    [
        (Search1APISearchBlock, "_search", "/search", "reddit"),
        (Search1APINewsBlock, "_news", "/news", "hackernews"),
    ],
)
async def test_search_blocks_forward_all_options(block_cls, method, endpoint, service):
    block = block_cls()
    captured: dict = {}

    def fake(_creds, payload):
        captured.update(payload)
        return {"results": [_row()]}

    _mock_block(block, {method: fake})
    await _collect(
        block,
        {
            "query": "agents",
            "search_service": service,
            "max_results": 7,
            "crawl_results": 2,
            "time_range": "week",
            "include_sites": ["github.com"],
            "exclude_sites": ["example.com"],
            "language": "en",
        },
    )

    assert captured == {
        "query": "agents",
        "search_service": service,
        "max_results": 7,
        "crawl_results": 2,
        "time_range": "week",
        "include_sites": ["github.com"],
        "exclude_sites": ["example.com"],
        "language": "en",
    }


@pytest.mark.asyncio
async def test_search_omits_unset_options():
    block = Search1APISearchBlock()
    captured: dict = {}

    def fake(_creds, payload):
        captured.update(payload)
        return {"results": []}

    _mock_block(block, {"_search": fake})
    await _collect(block, {"query": "agents"})

    assert captured == {"query": "agents", "max_results": 5, "crawl_results": 0}


@pytest.mark.asyncio
async def test_search_outputs_and_cost():
    block = Search1APISearchBlock()
    merged = _capture_stats(block)
    _mock_block(
        block,
        {
            "_search": lambda *a, **k: {
                "results": [
                    _row(content="full page"),
                    _row("Docs", "https://docs.agpt.co", content="docs page"),
                ]
            }
        },
    )

    out = await _collect(block, {"query": "AutoGPT", "crawl_results": 2})

    assert [r.url for r in out["results"][0]] == [
        "https://agpt.co",
        "https://docs.agpt.co",
    ]
    assert len(out["result"]) == 2
    assert "full page" in out["context"][0]
    # 1 credit for the search + 1 per crawled page.
    assert merged[0].provider_cost == pytest.approx(3 * CREDIT_USD)
    assert merged[0].provider_cost_type == "cost_usd"


@pytest.mark.asyncio
async def test_search_api_error_raises_block_error():
    block = Search1APISearchBlock()
    merged = _capture_stats(block)
    _mock_block(block, {"_search": _raise(ValueError("HTTP 402: no credits"))})

    with pytest.raises(BlockExecutionError, match="Search failed: HTTP 402"):
        await _collect(block, {"query": "AutoGPT"})
    assert merged == []


@pytest.mark.asyncio
async def test_news_keeps_published_date():
    block = Search1APINewsBlock()
    _mock_block(
        block,
        {"_news": lambda *a, **k: {"results": [_row(published_date="2026-10-04")]}},
    )

    out = await _collect(block, {"query": "AutoGPT"})

    assert out["result"][0].published_date == "2026-10-04"


def test_search_rejects_out_of_range_max_results():
    with pytest.raises(ValidationError):
        Search1APISearchBlock.Input.model_validate(
            {"credentials": TEST_CREDENTIALS_INPUT, "query": "q", "max_results": 51}
        )


@pytest.mark.parametrize(
    "block_cls, query_field",
    [
        (Search1APISearchBlock, {"query": "q"}),
        (Search1APINewsBlock, {"query": "q"}),
        (Search1APIBatchSearchBlock, {"queries": ["q"]}),
    ],
)
def test_crawl_results_cannot_exceed_max_results(block_cls, query_field):
    base = {"credentials": TEST_CREDENTIALS_INPUT, **query_field}
    block_cls.Input.model_validate({**base, "max_results": 3, "crawl_results": 3})
    with pytest.raises(ValidationError, match="crawl_results cannot be greater"):
        block_cls.Input.model_validate({**base, "max_results": 3, "crawl_results": 4})


# ---------------------------------------------------------------------------
# Crawl block
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_crawl_outputs_page_and_cost():
    block = Search1APICrawlBlock()
    merged = _capture_stats(block)
    captured: dict = {}

    def fake(_creds, url):
        captured["url"] = url
        return {
            "crawlParameters": {"url": url},
            "results": {
                "title": "AutoGPT",
                "link": "https://agpt.co/",
                "content": "# AutoGPT",
            },
        }

    _mock_block(block, {"_crawl": fake})
    out = await _collect(block, {"url": "https://agpt.co"})

    assert captured["url"] == "https://agpt.co"
    assert out["url"] == ["https://agpt.co/"]
    assert out["title"] == ["AutoGPT"]
    assert out["content"] == ["# AutoGPT"]
    assert merged[0].provider_cost == pytest.approx(CREDIT_USD)


@pytest.mark.asyncio
async def test_crawl_malformed_response_raises():
    for response in (
        {"results": []},
        {"results": {"title": "AutoGPT", "link": "https://agpt.co"}},
        {"results": {"title": "AutoGPT", "link": "https://agpt.co", "content": 1}},
    ):
        block = Search1APICrawlBlock()
        _mock_block(block, {"_crawl": lambda *a, _r=response, **k: _r})

        with pytest.raises(BlockExecutionError, match="Crawl failed: malformed"):
            await _collect(block, {"url": "https://agpt.co"})


# ---------------------------------------------------------------------------
# Batch search block
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_batch_sends_one_request_with_shared_options():
    block = Search1APIBatchSearchBlock()
    calls: list = []

    def fake(_creds, payload):
        calls.append(payload)
        return {
            "results": [
                {"success": True, "cost": 1, "data": {"results": [_row()]}}
                for _ in payload
            ],
            "summary": {"totalCost": len(payload)},
        }

    _mock_block(block, {"_batch_search": fake})
    await _collect(
        block,
        {
            "queries": ["a", "b"],
            "search_service": "github",
            "max_results": 3,
            "time_range": "month",
        },
    )

    assert len(calls) == 1
    assert calls[0] == [
        {
            "query": q,
            "search_service": "github",
            "max_results": 3,
            "crawl_results": 0,
            "time_range": "month",
        }
        for q in ("a", "b")
    ]


@pytest.mark.asyncio
async def test_batch_isolates_failed_queries_and_reports_total_cost():
    block = Search1APIBatchSearchBlock()
    merged = _capture_stats(block)
    _mock_block(
        block,
        {
            "_batch_search": lambda *a, **k: {
                "results": [
                    {"success": True, "cost": 1, "data": {"results": [_row()]}},
                    {
                        "success": False,
                        "cost": 0,
                        "error": {"message": "Upstream engine timed out"},
                    },
                    {"success": True, "cost": 3, "data": {"results": []}},
                ],
                "summary": {"total": 3, "successful": 2, "failed": 1, "totalCost": 4},
            }
        },
    )

    out = await _collect(block, {"queries": ["a", "b", "c"]})
    groups = out["results"][0]

    assert [g.query for g in groups] == ["a", "b", "c"]
    assert groups[0].results[0].url == "https://agpt.co" and groups[0].error == ""
    assert groups[1].results == [] and groups[1].error == "Upstream engine timed out"
    assert groups[2].results == [] and groups[2].error == ""
    assert merged[0].provider_cost == pytest.approx(4 * CREDIT_USD)

    context = out["context"][0]
    assert context.index("## Query: a") < context.index("## Query: b")
    assert "(https://agpt.co)" in context
    assert "## Query: b\n\nSearch failed: Upstream engine timed out" in context
    assert context.endswith("## Query: c\n\nNo results.")


@pytest.mark.asyncio
async def test_batch_cost_falls_back_to_item_costs():
    block = Search1APIBatchSearchBlock()
    merged = _capture_stats(block)
    _mock_block(
        block,
        {
            "_batch_search": lambda *a, **k: {
                "results": [
                    {"success": True, "cost": 2, "data": {"results": []}},
                    {"success": True, "cost": 1, "data": {"results": []}},
                ]
            }
        },
    )

    await _collect(block, {"queries": ["a", "b"]})

    assert merged[0].provider_cost == pytest.approx(3 * CREDIT_USD)


@pytest.mark.asyncio
async def test_batch_result_count_mismatch_raises():
    block = Search1APIBatchSearchBlock()
    _mock_block(block, {"_batch_search": lambda *a, **k: {"results": []}})

    with pytest.raises(BlockExecutionError, match="expected one result per query"):
        await _collect(block, {"queries": ["a"]})


def test_batch_rejects_more_than_ten_queries():
    with pytest.raises(ValidationError):
        Search1APIBatchSearchBlock.Input.model_validate(
            {"credentials": TEST_CREDENTIALS_INPUT, "queries": ["q"] * 11}
        )
