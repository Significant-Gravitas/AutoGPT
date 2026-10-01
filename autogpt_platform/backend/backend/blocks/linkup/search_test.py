"""Tests for LinkupSearchBlock.

Complements the ``test_mock`` harness exercised by ``test_available_blocks``
with coverage of the SDK-kwargs mapping, the per-output-type dispatch, cost
reporting, and error wrapping.
"""

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from linkup import (
    LinkupSearchImageResult,
    LinkupSearchResults,
    LinkupSearchTextResult,
    LinkupSource,
    LinkupSourcedAnswer,
)

from backend.blocks.linkup._api import search_cost_usd
from backend.blocks.linkup._config import linkup
from backend.blocks.linkup.search import LinkupSearchBlock
from backend.util.exceptions import BlockExecutionError

TEST_CREDENTIALS = linkup.get_test_credentials()


def _input(**kwargs) -> LinkupSearchBlock.Input:
    return LinkupSearchBlock.Input(
        credentials=TEST_CREDENTIALS.model_dump(),
        query="What is AutoGPT?",
        **kwargs,
    )


def _search_results() -> LinkupSearchResults:
    return LinkupSearchResults(
        results=[
            LinkupSearchTextResult(
                type="text",
                name="AutoGPT",
                url="https://agpt.co",
                content="AutoGPT is a platform for building AI agents.",
                favicon="",
            ),
            LinkupSearchImageResult(
                type="image", name="Logo", url="https://agpt.co/logo.png"
            ),
        ]
    )


def _sourced_answer() -> LinkupSourcedAnswer:
    return LinkupSourcedAnswer(
        answer="AutoGPT is a platform for building AI agents.",
        sources=[
            LinkupSource(
                name="AutoGPT",
                url="https://agpt.co",
                snippet="AutoGPT is a platform for building AI agents.",
                favicon="",
            )
        ],
    )


async def _run(
    block: LinkupSearchBlock, input_data: LinkupSearchBlock.Input
) -> dict[str, list]:
    outputs: dict[str, list] = {}
    async for name, value in block.run(input_data, credentials=TEST_CREDENTIALS):
        outputs.setdefault(name, []).append(value)
    return outputs


async def _run_and_capture_kwargs(input_data: LinkupSearchBlock.Input) -> dict:
    """Run the block with a mocked Linkup client and return the SDK call kwargs."""
    block = LinkupSearchBlock()
    with patch("backend.blocks.linkup.search.LinkupClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.async_search = AsyncMock(return_value=_search_results())
        mock_client_cls.return_value = mock_client

        await _run(block, input_data)

        return mock_client.async_search.call_args.kwargs


@pytest.mark.asyncio
async def test_run_maps_inputs_to_sdk_kwargs():
    kwargs = await _run_and_capture_kwargs(
        _input(
            depth="deep",
            max_results=3,
            include_domains=["agpt.co"],
            exclude_domains=["example.com"],
            from_date=date(2025, 1, 1),
            to_date=date(2025, 12, 31),
        )
    )

    assert kwargs == {
        "output_type": "searchResults",
        "query": "What is AutoGPT?",
        "depth": "deep",
        "max_results": 3,
        "include_domains": ["agpt.co"],
        "exclude_domains": ["example.com"],
        "from_date": date(2025, 1, 1),
        "to_date": date(2025, 12, 31),
    }


@pytest.mark.asyncio
async def test_run_omits_unset_filters():
    kwargs = await _run_and_capture_kwargs(_input())

    assert kwargs == {
        "output_type": "searchResults",
        "query": "What is AutoGPT?",
        "depth": "standard",
        "max_results": 10,
        "include_domains": None,
        "exclude_domains": None,
        "from_date": None,
        "to_date": None,
    }


@pytest.mark.asyncio
async def test_search_results_skips_image_results_and_reports_cost():
    block = LinkupSearchBlock()
    with (
        patch.object(
            block, "_search_results", AsyncMock(return_value=_search_results())
        ),
        patch.object(block, "merge_stats") as merge_stats,
    ):
        outputs = await _run(block, _input())

    [results] = outputs["results"]
    assert [r.url for r in results] == ["https://agpt.co"]
    assert [r.url for r in outputs["result"]] == ["https://agpt.co"]
    assert "[AutoGPT](https://agpt.co)" in outputs["context"][0]
    assert "answer" not in outputs

    stats = merge_stats.call_args.args[0]
    assert stats.provider_cost == pytest.approx(0.005)
    assert stats.provider_cost_type == "cost_usd"


@pytest.mark.asyncio
async def test_sourced_answer_yields_answer_and_sources():
    block = LinkupSearchBlock()
    search_results = AsyncMock()
    with (
        patch.object(block, "_search_results", search_results),
        patch.object(
            block, "_sourced_answer", AsyncMock(return_value=_sourced_answer())
        ),
        patch.object(block, "merge_stats") as merge_stats,
    ):
        outputs = await _run(block, _input(output_type="sourcedAnswer"))

    search_results.assert_not_awaited()
    assert outputs["answer"] == ["AutoGPT is a platform for building AI agents."]
    [sources] = outputs["sources"]
    assert sources[0].url == "https://agpt.co"
    assert "results" not in outputs
    assert merge_stats.call_args.args[0].provider_cost == pytest.approx(0.006)


@pytest.mark.asyncio
async def test_api_error_raises_block_error_without_cost():
    block = LinkupSearchBlock()
    with (
        patch.object(
            block,
            "_search_results",
            AsyncMock(side_effect=RuntimeError("upstream boom")),
        ),
        patch.object(block, "merge_stats") as merge_stats,
    ):
        with pytest.raises(BlockExecutionError, match="Search failed: upstream boom"):
            await _run(block, _input())

    merge_stats.assert_not_called()


@pytest.mark.parametrize(
    "depth, output_type, expected",
    [
        ("fast", "searchResults", 0.005),
        ("standard", "searchResults", 0.005),
        ("standard", "sourcedAnswer", 0.006),
        ("deep", "searchResults", 0.05),
        ("deep", "sourcedAnswer", 0.055),
    ],
)
def test_search_cost_usd(depth, output_type, expected):
    assert search_cost_usd(depth, output_type) == pytest.approx(expected)
