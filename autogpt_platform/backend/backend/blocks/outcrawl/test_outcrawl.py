"""Unit tests for the Outcrawl blocks.

Complements the ``test_mock`` harness in each block (exercised by
``test_available_blocks``) with direct coverage of the request payloads, the
credit-based cost reporting and the error boundaries.
"""

from unittest import mock

import pytest
from pydantic import SecretStr

from backend.blocks.outcrawl._api import (
    CREDIT_USD,
    OUTCRAWL_API_URL,
    OutcrawlClient,
    credits_from_response,
    result_from_dict,
)
from backend.blocks.outcrawl.scrape import OutcrawlScrapeBlock
from backend.blocks.outcrawl.search import OutcrawlSearchBlock
from backend.data.execution import ExecutionContext
from backend.sdk import APIKeyCredentials
from backend.util.exceptions import BlockExecutionError

TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="outcrawl",
    api_key=SecretStr("mock-outcrawl-api-key"),
    title="Mock Outcrawl API key",
    expires_at=None,
)
TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}


async def _collect(block, input_data: dict) -> dict:
    """Run ``block.execute`` and gather its yielded outputs into a dict."""
    outputs = {}
    async for name, value in block.execute(
        input_data,
        credentials=TEST_CREDENTIALS,
        execution_context=ExecutionContext(),
    ):
        outputs[name] = value
    return outputs


def _ok_response(body: dict) -> mock.Mock:
    resp = mock.Mock()
    resp.ok = True
    resp.json.return_value = body
    return resp


def test_credits_from_response_parses_numeric_string():
    assert credits_from_response({"usage": {"credits": "3"}}, estimate=1) == 3.0


def test_credits_from_response_falls_back_to_estimate():
    assert credits_from_response({}, estimate=2) == 2
    assert credits_from_response({"usage": {"credits": None}}, estimate=1) == 1


def test_result_from_dict_defaults_rank_to_position():
    r = result_from_dict({"url": "https://x.test"}, rank=4)
    assert r.title == ""
    assert r.url == "https://x.test"
    assert r.snippet == ""
    assert r.rank == 4
    assert r.markdown is None


@pytest.mark.asyncio
async def test_client_scrape_posts_to_scrape_endpoint():
    client = OutcrawlClient(TEST_CREDENTIALS)
    client.requests.post = mock.AsyncMock(
        return_value=_ok_response({"markdown": "# Hi"})
    )

    out = await client.scrape({"url": "https://agpt.co", "formats": ["markdown"]})

    assert out["markdown"] == "# Hi"
    args, kwargs = client.requests.post.call_args
    assert args[0] == f"{OUTCRAWL_API_URL}/v1/scrape"
    assert kwargs["json"]["url"] == "https://agpt.co"
    assert client.requests.extra_headers == {
        "Authorization": "Bearer mock-outcrawl-api-key"
    }


@pytest.mark.asyncio
async def test_client_non_ok_status_raises():
    client = OutcrawlClient(TEST_CREDENTIALS)
    resp = mock.Mock()
    resp.ok = False
    resp.status = 402
    client.requests.post = mock.AsyncMock(return_value=resp)

    with pytest.raises(ValueError, match="HTTP 402"):
        await client.search({"query": "q"})
    resp.json.assert_not_called()


@pytest.mark.asyncio
async def test_scrape_block_sends_formats_and_reports_cost():
    block = OutcrawlScrapeBlock()
    captured = {}

    async def fake_scrape(credentials, payload):
        captured.update(payload)
        return {
            "url": "https://agpt.co/",
            "statusCode": 200,
            "markdown": "AutoGPT",
            "links": ["https://agpt.co/docs"],
            "metadata": {"title": "AutoGPT"},
            "usage": {"credits": "1"},
        }

    block._scrape = fake_scrape
    with mock.patch.object(block, "merge_stats") as merge_stats:
        out = await _collect(
            block,
            {
                "credentials": TEST_CREDENTIALS_INPUT,
                "url": "https://agpt.co",
                "formats": ["markdown", "links"],
                "only_main_content": False,
            },
        )

    assert captured == {
        "url": "https://agpt.co",
        "formats": ["markdown", "links"],
        "onlyMainContent": False,
    }
    assert out["markdown"] == "AutoGPT"
    assert out["links"] == ["https://agpt.co/docs"]
    assert "html" not in out
    assert out["status_code"] == 200
    stats = merge_stats.call_args.args[0]
    assert stats.provider_cost == pytest.approx(CREDIT_USD)
    assert stats.provider_cost_type == "cost_usd"


@pytest.mark.asyncio
async def test_scrape_block_wraps_api_errors():
    block = OutcrawlScrapeBlock()

    async def failing(credentials, payload):
        raise ValueError("Outcrawl /v1/scrape request failed with HTTP 401")

    block._scrape = failing
    with pytest.raises(BlockExecutionError, match="HTTP 401"):
        await _collect(
            block, {"credentials": TEST_CREDENTIALS_INPUT, "url": "https://agpt.co"}
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code", ["login_redirect", "blocked_interstitial", "empty_capture"]
)
async def test_scrape_block_rejects_unusable_pages(code):
    block = OutcrawlScrapeBlock()

    async def warned(credentials, payload):
        return {
            "url": "https://github.com/login",
            "statusCode": 200,
            "markdown": "Sign in to GitHub",
            "metadata": {"title": "Sign in to GitHub"},
            "warnings": [{"code": code, "message": "not the page asked for"}],
            "usage": {"credits": "1"},
        }

    block._scrape = warned
    with (
        mock.patch.object(block, "merge_stats") as merge_stats,
        pytest.raises(BlockExecutionError, match=f"{code}: not the page"),
    ):
        await _collect(
            block,
            {
                "credentials": TEST_CREDENTIALS_INPUT,
                "url": "https://github.com/settings/profile",
            },
        )
    # The page was still fetched and billed, so the cost is recorded.
    assert merge_stats.called


@pytest.mark.asyncio
async def test_scrape_block_keeps_content_with_other_warnings():
    block = OutcrawlScrapeBlock()

    async def warned(credentials, payload):
        return {
            "url": "https://agpt.co/",
            "statusCode": 200,
            "markdown": "AutoGPT",
            "metadata": {"title": "AutoGPT"},
            "warnings": [{"code": "some_other_code", "message": "informational"}],
            "usage": {"credits": "1"},
        }

    block._scrape = warned
    with mock.patch.object(block, "merge_stats"):
        out = await _collect(
            block, {"credentials": TEST_CREDENTIALS_INPUT, "url": "https://agpt.co"}
        )
    assert out["markdown"] == "AutoGPT"


@pytest.mark.asyncio
async def test_search_block_scrape_results_requests_markdown():
    block = OutcrawlSearchBlock()
    captured = {}

    async def fake_search(credentials, payload):
        captured.update(payload)
        return {
            "hits": [
                {
                    "title": "AutoGPT",
                    "url": "https://agpt.co",
                    "snippet": "snippet",
                    "rank": 1,
                    "markdown": "# AutoGPT page",
                },
                {"title": "Docs", "url": "https://docs.agpt.co", "rank": 2},
            ],
            "usage": {"credits": "3"},
        }

    block._search = fake_search
    with mock.patch.object(block, "merge_stats") as merge_stats:
        out = await _collect(
            block,
            {
                "credentials": TEST_CREDENTIALS_INPUT,
                "query": "autogpt",
                "limit": 2,
                "scrape_results": True,
            },
        )

    assert captured == {
        "query": "autogpt",
        "limit": 2,
        "scrape": True,
        "formats": ["markdown"],
    }
    assert [r.url for r in out["results"]] == [
        "https://agpt.co",
        "https://docs.agpt.co",
    ]
    assert out["results"][0].markdown == "# AutoGPT page"
    assert "# AutoGPT page" in out["context"]
    stats = merge_stats.call_args.args[0]
    assert stats.provider_cost == pytest.approx(3 * CREDIT_USD)


@pytest.mark.asyncio
async def test_search_block_without_scrape_sends_plain_query():
    block = OutcrawlSearchBlock()
    captured = {}

    async def fake_search(credentials, payload):
        captured.update(payload)
        return {"hits": []}

    block._search = fake_search
    with mock.patch.object(block, "merge_stats") as merge_stats:
        out = await _collect(
            block, {"credentials": TEST_CREDENTIALS_INPUT, "query": "autogpt"}
        )

    assert captured == {"query": "autogpt", "limit": 5}
    assert out["results"] == []
    assert out["context"] == ""
    # No usage report: falls back to the documented 1 credit per search.
    assert merge_stats.call_args.args[0].provider_cost == pytest.approx(CREDIT_USD)
