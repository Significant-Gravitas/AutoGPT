"""Tests for LinkupFetchBlock: SDK-kwargs mapping, cost reporting and error
wrapping, complementing the ``test_mock`` harness in the block itself."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from linkup import LinkupFetchResponse

from backend.blocks.linkup._config import linkup
from backend.blocks.linkup.fetch import LinkupFetchBlock
from backend.util.exceptions import BlockExecutionError

TEST_CREDENTIALS = linkup.get_test_credentials()


def _input(**kwargs) -> LinkupFetchBlock.Input:
    return LinkupFetchBlock.Input(
        credentials=TEST_CREDENTIALS.model_dump(),
        url="https://agpt.co",
        **kwargs,
    )


async def _run(block: LinkupFetchBlock, input_data: LinkupFetchBlock.Input) -> dict:
    return {
        name: value
        async for name, value in block.run(input_data, credentials=TEST_CREDENTIALS)
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("render_js, expected_cost", [(False, 0.001), (True, 0.005)])
async def test_fetch_returns_markdown_and_reports_cost(render_js, expected_cost):
    block = LinkupFetchBlock()
    with (
        patch("backend.blocks.linkup.fetch.LinkupClient") as mock_client_cls,
        patch.object(block, "merge_stats") as merge_stats,
    ):
        mock_client = MagicMock()
        mock_client.async_fetch = AsyncMock(
            return_value=LinkupFetchResponse(markdown="# AutoGPT", favicon="")
        )
        mock_client_cls.return_value = mock_client

        outputs = await _run(block, _input(render_js=render_js))

    assert outputs == {"markdown": "# AutoGPT"}
    assert mock_client.async_fetch.call_args.kwargs == {
        "url": "https://agpt.co",
        "render_js": render_js,
    }
    stats = merge_stats.call_args.args[0]
    assert stats.provider_cost == pytest.approx(expected_cost)
    assert stats.provider_cost_type == "cost_usd"


@pytest.mark.asyncio
async def test_fetch_api_error_raises_block_error_without_cost():
    block = LinkupFetchBlock()
    with (
        patch.object(
            block, "_fetch", AsyncMock(side_effect=RuntimeError("upstream boom"))
        ),
        patch.object(block, "merge_stats") as merge_stats,
    ):
        with pytest.raises(BlockExecutionError, match="Fetch failed: upstream boom"):
            await _run(block, _input())

    merge_stats.assert_not_called()
