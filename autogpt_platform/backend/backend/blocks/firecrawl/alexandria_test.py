"""Unit tests for the Firecrawl Alexandria blocks."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks.firecrawl.alexandria import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    AlexandriaError,
    FirecrawlFindDataToolsBlock,
    FirecrawlRunDataToolBlock,
    call_alexandria,
)
from backend.data.model import NodeExecutionStats


def _response(status: int, body: dict) -> MagicMock:
    response = MagicMock()
    response.status = status
    response.ok = 200 <= status < 300
    response.json.return_value = body
    return response


def _patch_post(response: MagicMock):
    requests = MagicMock()
    requests.post = AsyncMock(return_value=response)
    return patch("backend.blocks.firecrawl.alexandria.Requests", return_value=requests)


async def test_call_alexandria_sends_one_call_with_auth_and_request_id():
    body = {
        "success": True,
        "data": {
            "alexandria": [{"provider": "apollo", "data": [], "creditsCost": 2}],
            "creditsCost": 2,
        },
    }
    with _patch_post(_response(200, body)) as requests_cls:
        result = await call_alexandria(
            "fc-key", "apollo", "people/search", {"titles": ["CRO"]}, 300
        )

    assert result == body["data"]["alexandria"][0]
    post = requests_cls.return_value.post
    url = post.call_args.args[0]
    kwargs = post.call_args.kwargs
    assert url == "https://api.firecrawl.dev/v2/scrape"
    assert kwargs["headers"]["Authorization"] == "Bearer fc-key"
    assert kwargs["headers"]["x-request-id"]
    assert kwargs["json"] == {
        "alexandria": [
            {
                "provider": "apollo",
                "capability": "people/search",
                "options": {"titles": ["CRO"]},
            }
        ],
        # Capped at Firecrawl's 120-second maximum, sent in milliseconds.
        "timeout": 120_000,
    }


async def test_call_alexandria_points_to_terms_page_when_terms_are_required():
    body = {
        "success": False,
        "code": "THIRD_PARTY_DATA_TERMS_REQUIRED",
        "error": "An organization admin must accept the apollo provider's terms",
        "requiresAction": {"url": "https://www.firecrawl.dev/app/alexandria/apollo"},
    }
    with _patch_post(_response(403, body)), pytest.raises(AlexandriaError) as caught:
        await call_alexandria("fc-key", "apollo", "people/search")

    assert "https://www.firecrawl.dev/app/alexandria/apollo" in str(caught.value)


async def test_call_alexandria_reports_other_refusals_with_their_code():
    body = {"success": False, "code": "INSUFFICIENT_CREDITS", "error": "No credits"}
    with (
        _patch_post(_response(402, body)),
        pytest.raises(AlexandriaError, match="INSUFFICIENT_CREDITS: No credits"),
    ):
        await call_alexandria("fc-key", "apollo", "people/search")


async def _run(block, **inputs):
    captured: list[NodeExecutionStats] = []
    outputs: dict = {}
    with patch.object(block, "merge_stats", side_effect=captured.append):
        async for name, value in block.run(
            block.input_schema(credentials=TEST_CREDENTIALS_INPUT, **inputs),
            credentials=TEST_CREDENTIALS,
        ):
            outputs.setdefault(name, []).append(value)
    return outputs, captured


async def test_run_data_tool_bills_the_credits_firecrawl_reports():
    block = FirecrawlRunDataToolBlock()
    result = {"creditsCost": 40, "records": 2, "data": [{"a": 1}, {"a": 2}]}
    with patch.object(block, "call_alexandria", AsyncMock(return_value=result)):
        outputs, stats = await _run(block, provider="apollo", capability="x")

    assert outputs["records"] == [2]
    assert outputs["credits_cost"] == [40.0]
    assert stats[0].provider_cost == pytest.approx(0.04)
    assert stats[0].provider_cost_type == "cost_usd"


async def test_run_data_tool_bills_then_raises_when_the_capability_failed():
    block = FirecrawlRunDataToolBlock()
    result = {
        "creditsCost": 3,
        "error": {"code": "invalid_options", "message": "titles is required"},
    }
    captured: list[NodeExecutionStats] = []
    with (
        patch.object(block, "call_alexandria", AsyncMock(return_value=result)),
        patch.object(block, "merge_stats", side_effect=captured.append),
        pytest.raises(AlexandriaError, match="invalid_options: titles is required"),
    ):
        async for _ in block.run(
            block.input_schema(
                credentials=TEST_CREDENTIALS_INPUT, provider="apollo", capability="x"
            ),
            credentials=TEST_CREDENTIALS,
        ):
            pass

    # A failed capability can still be charged, so its cost is recorded.
    assert captured[0].provider_cost == pytest.approx(0.003)


@pytest.mark.parametrize(
    "inputs, expected",
    [
        (
            {"query": "work emails"},
            {
                "level": "tools",
                "limit": 10,
                "query": "work emails",
                "expand": ["options", "response"],
            },
        ),
        (
            {"providers": ["apollo"], "include_contracts": False},
            {"level": "tools", "limit": 10, "providers": ["apollo"], "expand": []},
        ),
        (
            {"categories": ["people"], "offset": 20},
            {"level": "providers", "limit": 10, "offset": 20, "categories": ["people"]},
        ),
        ({}, {"level": "categories", "limit": 10}),
    ],
)
def test_find_data_tools_builds_catalogue_options(inputs, expected):
    input_data = FirecrawlFindDataToolsBlock.Input(
        credentials=TEST_CREDENTIALS_INPUT, **inputs  # type: ignore[arg-type]
    )
    assert FirecrawlFindDataToolsBlock.build_options(input_data) == expected
