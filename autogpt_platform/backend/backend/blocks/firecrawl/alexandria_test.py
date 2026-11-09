"""Unit tests for the Firecrawl Alexandria blocks."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks.firecrawl._config import firecrawl
from backend.blocks.firecrawl.alexandria import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    AlexandriaError,
    FirecrawlFindDataToolsBlock,
    FirecrawlRunDataToolBlock,
    call_alexandria,
)
from backend.data.block_cost_config import BLOCK_COSTS
from backend.data.model import NodeExecutionStats
from backend.executor.billing import _block_has_paid_cost_entry
from backend.sdk.cost_integration import register_provider_costs_for_block
from backend.util.exceptions import BlockInputError


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


async def test_call_alexandria_bounds_retries_and_waiting():
    body = {"success": True, "data": {"alexandria": [{"creditsCost": 0}]}}
    with _patch_post(_response(200, body)) as requests_cls:
        await call_alexandria("fc-key", "apollo", "people/search")

    # A throttled or failing Firecrawl must not hold an executor worker.
    assert requests_cls.call_args.kwargs["retry_max_attempts"] == 3
    timeout = requests_cls.return_value.post.call_args.kwargs["timeout"]
    assert timeout.total == 150


async def test_call_alexandria_reuses_a_given_request_id():
    body = {"success": True, "data": {"alexandria": [{"creditsCost": 0}]}}
    with _patch_post(_response(200, body)) as requests_cls:
        await call_alexandria("fc-key", "apollo", "x", request_id=" req-1 ")

    headers = requests_cls.return_value.post.call_args.kwargs["headers"]
    assert headers["x-request-id"] == "req-1"


async def test_unresolved_call_names_the_request_id_to_retry_with():
    body = {"success": False, "code": "request_unresolved", "error": "Unknown"}
    with (
        _patch_post(_response(503, body)) as requests_cls,
        pytest.raises(AlexandriaError) as caught,
    ):
        await call_alexandria("fc-key", "apollo", "x")

    sent = requests_cls.return_value.post.call_args.kwargs["headers"]["x-request-id"]
    assert f"run again with request ID {sent}" in str(caught.value)


@pytest.mark.parametrize(
    "body",
    [
        {"success": True, "data": {"alexandria": []}},
        {"success": True, "data": None},
    ],
)
async def test_call_alexandria_raises_when_there_is_no_result(body: dict[str, Any]):
    with (
        _patch_post(_response(200, body)),
        pytest.raises(AlexandriaError, match="no Alexandria result"),
    ):
        await call_alexandria("fc-key", "apollo", "x")


async def test_failed_call_names_the_request_id_to_retry_with():
    requests = MagicMock()
    requests.post = AsyncMock(side_effect=TimeoutError())
    with (
        patch("backend.blocks.firecrawl.alexandria.Requests", return_value=requests),
        pytest.raises(AlexandriaError) as caught,
    ):
        await call_alexandria("fc-key", "apollo", "x", request_id="req-9")

    assert "run again with request ID req-9" in str(caught.value)


def test_refusals_are_user_fixable_errors():
    # The executor treats ValueErrors as the user's to fix, not platform faults.
    assert issubclass(AlexandriaError, ValueError)


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


async def test_run_data_tool_sends_trimmed_provider_and_capability():
    block = FirecrawlRunDataToolBlock()
    call = AsyncMock(return_value={"creditsCost": 0, "records": 0, "data": []})
    with patch.object(block, "call_alexandria", call):
        await _run(block, provider="  apollo ", capability=" people/search  ")

    assert call.call_args.args[1:3] == ("apollo", "people/search")


@pytest.mark.parametrize(
    "result, records",
    [
        ({"data": [{"a": 1}, {"a": 2}]}, 2),
        ({"data": {"people": []}}, 1),
        ({"data": None}, 0),
        ({"data": [], "records": 3.0}, 3),
        ({"data": [{"a": 1}], "records": "many"}, 1),
    ],
)
async def test_run_data_tool_counts_records(result: dict[str, Any], records: int):
    block = FirecrawlRunDataToolBlock()
    with patch.object(
        block, "call_alexandria", AsyncMock(return_value={"creditsCost": 0, **result})
    ):
        outputs, _ = await _run(block, provider="apollo", capability="x")

    assert outputs["records"] == [records]


@pytest.mark.parametrize("provider, capability", [(" ", "x"), ("apollo", " ")])
async def test_run_data_tool_rejects_blank_provider_or_capability(
    provider: str, capability: str
):
    block = FirecrawlRunDataToolBlock()
    call = AsyncMock()
    with (
        patch.object(block, "call_alexandria", call),
        pytest.raises(BlockInputError, match="Provider and capability"),
    ):
        await _run(block, provider=provider, capability=capability)
    call.assert_not_called()


def test_find_data_tools_is_free_on_the_platform_key():
    """Discovery must not inherit Firecrawl's paid provider cost: a paid entry
    stops the block on an empty wallet even though it charges nothing."""
    platform_input = {"credentials": {"id": TEST_CREDENTIALS.id}}
    with (
        patch.object(firecrawl, "default_credentials", [TEST_CREDENTIALS]),
        patch.dict(BLOCK_COSTS),
    ):
        BLOCK_COSTS.pop(FirecrawlRunDataToolBlock, None)
        for block_class in (FirecrawlFindDataToolsBlock, FirecrawlRunDataToolBlock):
            register_provider_costs_for_block(block_class)

        assert not _block_has_paid_cost_entry(
            FirecrawlFindDataToolsBlock(), platform_input
        )
        assert _block_has_paid_cost_entry(FirecrawlRunDataToolBlock(), platform_input)


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
def test_find_data_tools_builds_catalogue_options(
    inputs: dict[str, Any], expected: dict[str, Any]
):
    input_data = FirecrawlFindDataToolsBlock.Input(
        credentials=TEST_CREDENTIALS_INPUT, **inputs
    )
    assert FirecrawlFindDataToolsBlock.build_options(input_data) == expected
