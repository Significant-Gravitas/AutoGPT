"""Tests for ExaContentsBlock.run() SDK-kwargs mapping.

Covers the ``output_schema`` → ``"schema"`` remap and the int-count omission
behaviour that the ``run`` method builds inline (separate from the
``process_contents_settings`` path tested in ``helpers_cost_test.py``).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks.exa._test import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.exa.contents import ExaContentsBlock
from backend.blocks.exa.helpers import ExtrasSettings, SummarySettings


def _mock_response() -> MagicMock:
    response = MagicMock()
    response.results = []
    response.context = None
    response.statuses = None
    response.cost_dollars = None
    return response


async def _run_and_capture_kwargs(input_data: ExaContentsBlock.Input) -> dict:
    """Run the block with a mocked Exa client and return the SDK call kwargs."""
    block = ExaContentsBlock()
    with patch("backend.blocks.exa.contents.AsyncExa") as mock_exa_cls:
        mock_client = MagicMock()
        mock_client.get_contents = AsyncMock(return_value=_mock_response())
        mock_exa_cls.return_value = mock_client

        async for _ in block.run(input_data, credentials=TEST_CREDENTIALS):
            pass

        return mock_client.get_contents.call_args.kwargs


@pytest.mark.asyncio
async def test_run_maps_output_schema_to_schema_kwarg():
    """``summary.output_schema`` must reach the SDK as ``summary={"schema": ...}``."""
    input_data = ExaContentsBlock.Input(
        credentials=TEST_CREDENTIALS_INPUT,
        urls=["https://example.com"],
        summary=SummarySettings(query="overview", output_schema={"type": "object"}),
    )

    kwargs = await _run_and_capture_kwargs(input_data)

    assert kwargs["summary"] == {"query": "overview", "schema": {"type": "object"}}
    assert "output_schema" not in kwargs["summary"]


@pytest.mark.asyncio
async def test_run_omits_zero_extras():
    """Zero-valued int counts must be omitted, not sent as ``0`` to the API."""
    input_data = ExaContentsBlock.Input(
        credentials=TEST_CREDENTIALS_INPUT,
        urls=["https://example.com"],
        extras=ExtrasSettings(links=0, image_links=3),
    )

    kwargs = await _run_and_capture_kwargs(input_data)

    assert kwargs["extras"] == {"image_links": 3}
    assert "links" not in kwargs["extras"]
