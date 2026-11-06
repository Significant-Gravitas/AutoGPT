"""Tests for the SpecStream Create Dataset Request block."""

from typing import Any
from unittest.mock import AsyncMock

import pytest

from backend.blocks.specstream._config import TEST_CREDENTIALS
from backend.blocks.specstream._models import Reading, UnsupportedClause
from backend.blocks.specstream._test_helpers import run_block
from backend.blocks.specstream._testdata import TEST_AST_HASH, TEST_QUOTE, TEST_READING
from backend.blocks.specstream.dataset_request import (
    SpecStreamCreateDatasetRequestBlock,
)
from backend.util.exceptions import BlockExecutionError, BlockInputError


@pytest.mark.asyncio
async def test_create_request_refuses_an_unrunnable_reading_before_pricing():
    needs_edit = Reading(
        status="needs_edit",
        ast_hash=TEST_AST_HASH,
        unsupported=[UnsupportedClause(field_id="x", reason="No source covers this.")],
    )
    block = SpecStreamCreateDatasetRequestBlock()
    block.read_request = AsyncMock(return_value=needs_edit)
    block.price_reading = AsyncMock()

    with pytest.raises(BlockInputError, match="No source covers this"):
        await run_block(block, request="anything")

    block.price_reading.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_request_says_when_the_service_is_down():
    block = SpecStreamCreateDatasetRequestBlock()
    block.read_request = AsyncMock(return_value=Reading(status="compiler_unavailable"))
    block.price_reading = AsyncMock()

    with pytest.raises(BlockExecutionError, match="service is down"):
        await run_block(block, request="anything")

    block.price_reading.assert_not_awaited()


@pytest.mark.parametrize(
    "limit, expected",
    [
        ({"max": 100, "inferred": False}, 100),
        ({"max": 25, "inferred": True}, 40),
        ({"max": 50000, "inferred": False}, 10_000),
    ],
)
@pytest.mark.asyncio
async def test_create_request_takes_the_row_count_from_the_request(
    limit: dict[str, Any], expected: int
):
    block = SpecStreamCreateDatasetRequestBlock()
    block.read_request = AsyncMock(
        return_value=Reading.model_validate(
            {"status": "runnable", "ast_hash": TEST_AST_HASH, "ast": {"limit": limit}}
        )
    )
    block.price_reading = AsyncMock(return_value=TEST_QUOTE)

    await run_block(block, request="100 fintech startups", row_count=40)

    block.price_reading.assert_awaited_once_with(
        TEST_CREDENTIALS, TEST_AST_HASH, expected
    )


@pytest.mark.asyncio
async def test_create_request_summary_states_the_maximum():
    block = SpecStreamCreateDatasetRequestBlock()
    block.read_request = AsyncMock(return_value=TEST_READING)
    block.price_reading = AsyncMock(return_value=TEST_QUOTE)

    outputs = await run_block(block, request="fintech startups", row_count=25)

    block.price_reading.assert_awaited_once_with(TEST_CREDENTIALS, TEST_AST_HASH, 25)
    assert outputs["maximum_cents"] == 375
    assert outputs["price_summary"].startswith("Up to 3.75 USD for up to 25 rows")
