"""Unit tests for the Keenable blocks.

Complements the ``test_mock`` harness in each block (exercised by
``test_available_blocks``) with direct coverage of the client's keyless and
keyed endpoint selection, response mapping and error boundaries.
"""

from unittest import mock

import pytest
from pydantic import SecretStr

from backend.blocks.keenable._api import (
    KEENABLE_API_URL,
    KeenableAuth,
    KeenableClient,
    results_from_response,
)
from backend.blocks.keenable._config import keenable
from backend.blocks.keenable.fetch import KeenableFetchBlock
from backend.blocks.keenable.search import KeenableSearchBlock
from backend.data.block_cost_config import BLOCK_COSTS
from backend.data.execution import ExecutionContext
from backend.executor.utils import block_usage_cost
from backend.sdk import APIKeyCredentials
from backend.sdk.cost_integration import register_provider_costs_for_block
from backend.util.exceptions import BlockExecutionError

KEENABLE_PLATFORM_KEY = APIKeyCredentials(
    id="keenable-default",
    provider="keenable",
    api_key=SecretStr("platform-key"),
    title="Keenable API Key",
)

TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="keenable",
    api_key=SecretStr("mock-keenable-api-key"),
    title="Mock Keenable API key",
    expires_at=None,
)


def _ok(payload: dict) -> mock.Mock:
    resp = mock.Mock()
    resp.ok = True
    resp.json.return_value = payload
    return resp


async def _collect(block, input_data: dict, credentials=None) -> dict:
    outputs = {}
    async for name, value in block.execute(
        input_data, credentials=credentials, execution_context=ExecutionContext()
    ):
        outputs[name] = value
    return outputs


def test_results_from_response_maps_and_skips_rows_without_url():
    results = results_from_response(
        {
            "results": [
                {
                    "title": "A",
                    "url": "https://a.test",
                    "description": "",
                    "snippet": "long excerpt",
                    "published_at": "2026-05-06T10:00:00Z",
                },
                {"title": "no url", "url": "", "snippet": "dropped"},
                {"url": "https://b.test", "description": "fallback"},
            ]
        }
    )
    assert [r.url for r in results] == ["https://a.test", "https://b.test"]
    assert results[0].snippet == "long excerpt"
    assert results[0].published_at == "2026-05-06T10:00:00Z"
    assert results[1].title == ""
    assert results[1].snippet == "fallback"


def test_results_from_response_rejects_malformed_payload():
    with pytest.raises(TypeError, match="no results list"):
        results_from_response({"detail": "oops"})


@pytest.mark.asyncio
async def test_keyless_client_uses_public_endpoints_and_app_title():
    client = KeenableClient(None)
    assert client.requests.extra_headers == {"X-Keenable-Title": "AutoGPT"}
    client.requests.post = mock.AsyncMock(return_value=_ok({"results": []}))
    client.requests.get = mock.AsyncMock(return_value=_ok({"content": "x"}))

    await client.search({"query": "q", "max_results": 1})
    args, kwargs = client.requests.post.call_args
    assert args[0] == f"{KEENABLE_API_URL}/v1/search/public"
    assert kwargs["json"] == {"query": "q", "max_results": 1}

    await client.fetch("https://a.test/x?y=1", max_chars=500)
    (url,), _ = client.requests.get.call_args
    assert url == (
        f"{KEENABLE_API_URL}/v1/fetch/public"
        "?url=https%3A%2F%2Fa.test%2Fx%3Fy%3D1&live=true&max_chars=500"
    )


@pytest.mark.asyncio
async def test_keyed_client_uses_keyed_endpoints_and_api_key_header():
    client = KeenableClient(TEST_CREDENTIALS)
    assert client.requests.extra_headers == {"X-API-Key": "mock-keenable-api-key"}
    assert "X-API-Key" in client.requests.sensitive_headers
    client.requests.post = mock.AsyncMock(return_value=_ok({"results": []}))
    await client.search({"query": "q"})
    args, _ = client.requests.post.call_args
    assert args[0] == f"{KEENABLE_API_URL}/v1/search"


def test_empty_api_key_falls_back_to_keyless():
    creds = TEST_CREDENTIALS.model_copy(update={"api_key": SecretStr("  ")})
    client = KeenableClient(creds)
    assert client.keyed is False
    assert client.requests.extra_headers == {"X-Keenable-Title": "AutoGPT"}


@pytest.mark.asyncio
async def test_client_non_ok_status_raises():
    client = KeenableClient(None)
    resp = mock.Mock()
    resp.ok = False
    resp.status = 302
    client.requests.post = mock.AsyncMock(return_value=resp)
    with pytest.raises(ValueError, match="HTTP 302"):
        await client.search({"query": "q"})


@pytest.mark.asyncio
async def test_search_block_anonymous_ignores_selected_credentials_and_sends_filters():
    block = KeenableSearchBlock()
    seen = {}

    async def fake_search(credentials, payload):
        seen["credentials"] = credentials
        seen["payload"] = payload
        return {
            "results": [{"title": "A", "url": "https://a.test", "snippet": "excerpt"}]
        }

    with mock.patch.object(block, "_search", fake_search):
        out = await _collect(
            block,
            {
                "auth": KeenableAuth.ANONYMOUS.value,
                "query": "q",
                "max_results": 3,
                "site": "a.test",
                "published_after": "7d",
            },
            credentials=TEST_CREDENTIALS,
        )
        assert seen["credentials"] is None
        assert seen["payload"] == {
            "query": "q",
            "max_results": 3,
            "site": "a.test",
            "published_after": "7d",
        }
        assert out["result"].url == "https://a.test"
        assert out["context"] == "[A](https://a.test)\nexcerpt"


@pytest.mark.asyncio
async def test_search_block_wraps_upstream_errors():
    block = KeenableSearchBlock()

    async def boom(*args, **kwargs):
        raise ValueError("Keenable search request failed with HTTP 429")

    with (
        mock.patch.object(block, "_search", boom),
        pytest.raises(BlockExecutionError, match="HTTP 429"),
    ):
        await _collect(block, {"query": "q"})


@pytest.mark.asyncio
async def test_fetch_block_reports_truncation():
    block = KeenableFetchBlock()

    async def fake_fetch(credentials, url, max_chars):
        assert credentials is None
        assert max_chars == 100
        return {
            "url": url,
            "title": "T",
            "content": "# T\n\n..._This content has been truncated to stay below 100 characters_...",
        }

    with mock.patch.object(block, "_fetch", fake_fetch):
        out = await _collect(block, {"url": "https://a.test", "max_chars": 100})
        assert out["title"] == "T"
        assert out["truncated"] is True


@pytest.mark.asyncio
async def test_fetch_block_rejects_malformed_fields():
    block = KeenableFetchBlock()

    async def fake_fetch(*args, **kwargs):
        return {"url": "https://a.test", "title": 1, "content": "x"}

    with (
        mock.patch.object(block, "_fetch", fake_fetch),
        pytest.raises(BlockExecutionError, match="malformed"),
    ):
        await _collect(block, {"url": "https://a.test"})


@pytest.mark.asyncio
async def test_fetch_block_ignores_marker_text_quoted_inside_the_page():
    block = KeenableFetchBlock()

    async def fake_fetch(*args, **kwargs):
        return {
            "url": "https://a.test",
            "title": "T",
            "content": "_This content has been truncated_ is a phrase.\n\n" + "x" * 500,
        }

    with mock.patch.object(block, "_fetch", fake_fetch):
        out = await _collect(block, {"url": "https://a.test"})
    assert out["truncated"] is False


@pytest.mark.asyncio
async def test_fetch_block_rejects_missing_content():
    block = KeenableFetchBlock()

    async def fake_fetch(*args, **kwargs):
        return {"url": "https://a.test", "title": "T"}

    with (
        mock.patch.object(block, "_fetch", fake_fetch),
        pytest.raises(BlockExecutionError, match="malformed"),
    ):
        await _collect(block, {"url": "https://a.test"})


@pytest.mark.asyncio
async def test_search_block_escapes_markdown_links():
    block = KeenableSearchBlock()

    async def fake_search(credentials, payload):
        return {
            "results": [
                {"title": "A [b]", "url": "https://a.test/x(1)", "snippet": "s"}
            ]
        }

    with mock.patch.object(block, "_search", fake_search):
        out = await _collect(
            block, {"auth": KeenableAuth.ANONYMOUS.value, "query": "q"}
        )
    assert out["context"] == "[A \\[b\\]](https://a.test/x%281%29)\ns"


@pytest.mark.asyncio
async def test_fetch_block_rejects_falsy_non_string_title():
    block = KeenableFetchBlock()

    async def fake_fetch(*args, **kwargs):
        return {"url": "https://a.test", "title": 0, "content": "x"}

    with (
        mock.patch.object(block, "_fetch", fake_fetch),
        pytest.raises(BlockExecutionError, match="malformed"),
    ):
        await _collect(block, {"url": "https://a.test"})


@pytest.mark.asyncio
async def test_search_block_rejects_blank_key_in_api_key_mode():
    blank = TEST_CREDENTIALS.model_copy(update={"api_key": SecretStr("   ")})
    with pytest.raises(Exception, match="credentials are required"):
        await _collect(
            KeenableSearchBlock(),
            {"auth": KeenableAuth.API_KEY.value, "query": "q"},
            credentials=blank,
        )


def test_platform_cost_only_applies_in_api_key_mode():
    block_class = KeenableSearchBlock
    saved_costs = BLOCK_COSTS.get(block_class)
    saved_default_creds = keenable.default_credentials
    BLOCK_COSTS.pop(block_class, None)
    keenable.default_credentials = [KEENABLE_PLATFORM_KEY]
    register_provider_costs_for_block(block_class)
    try:
        block = block_class()
        platform_key = {
            "auth": KeenableAuth.API_KEY.value,
            "credentials": {
                "id": KEENABLE_PLATFORM_KEY.id,
                "provider": "keenable",
            },
        }
        assert block_usage_cost(block, platform_key)[0] == 1
        assert block_usage_cost(block, {"auth": KeenableAuth.ANONYMOUS.value})[0] == 0
        assert block_usage_cost(block, {})[0] == 0
        assert all(
            entry.cost_filter.get("auth") == KeenableAuth.API_KEY.value
            for entry in BLOCK_COSTS[block_class]
            if not entry.is_free
        )
    finally:
        keenable.default_credentials = saved_default_creds
        if saved_costs is None:
            BLOCK_COSTS.pop(block_class, None)
        else:
            BLOCK_COSTS[block_class] = saved_costs
