"""Tests for the SpecStream Approve And Run block."""

from unittest.mock import AsyncMock, patch

import pytest

from backend.blocks.specstream._config import TEST_CREDENTIALS_INPUT
from backend.blocks.specstream._test_helpers import (
    QUOTE_BODY,
    FakeResponse,
    approve_inputs,
    execution,
    fake_requests,
    run_block,
)
from backend.blocks.specstream._testdata import TEST_QUOTE_HASH
from backend.blocks.specstream.approve_and_run import SpecStreamApproveAndRunBlock
from backend.util.exceptions import BlockExecutionError


@pytest.mark.parametrize(
    "key, expected",
    [(None, "autogpt-qte_test_001"), ("my-own-key-1", "my-own-key-1")],
)
@pytest.mark.asyncio
async def test_approve_uses_a_key_that_a_retry_repeats(key: str | None, expected: str):
    block = SpecStreamApproveAndRunBlock()
    block.authorize = AsyncMock(return_value=execution())

    await run_block(block, **approve_inputs(idempotency_key=key))

    assert block.authorize.await_args.args[2] == expected


def test_approve_is_an_irreversible_action_that_needs_a_stated_maximum():
    block = SpecStreamApproveAndRunBlock()
    assert block.is_irreversible_action
    with pytest.raises(ValueError):
        block.input_schema(
            credentials=TEST_CREDENTIALS_INPUT,
            quote_id="qte_1",
            quote_hash=TEST_QUOTE_HASH,
        )
    with pytest.raises(ValueError):
        block.input_schema(
            credentials=TEST_CREDENTIALS_INPUT,
            **approve_inputs(idempotency_key="short"),
        )


@pytest.mark.asyncio
async def test_approve_reads_the_run_id_from_the_quote_when_authorize_has_none():
    requests = fake_requests(
        FakeResponse(
            202,
            {
                "data": {
                    "reservation_id": "rsv_1",
                    "maximum_cents": 375,
                    "state": "reserved",
                }
            },
        ),
        FakeResponse(
            200,
            {"data": {**QUOTE_BODY, "existing_execution": {"find_id": "fnd_9"}}},
        ),
    )
    with patch("backend.blocks.specstream._api.Requests", return_value=requests):
        outputs = await run_block(SpecStreamApproveAndRunBlock(), **approve_inputs())

    assert outputs["run_id"] == "fnd_9"


@pytest.mark.asyncio
async def test_approve_names_the_reservation_when_no_run_id_comes_back():
    block = SpecStreamApproveAndRunBlock()
    block.authorize = AsyncMock(return_value=execution(find_id=None))

    with pytest.raises(BlockExecutionError, match="rsv_test_001"):
        await run_block(block, **approve_inputs())
