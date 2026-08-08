"""Tests for SearchFilterTypeaheadBlock.run() input validation and limit clamping."""

import json
from unittest.mock import AsyncMock, patch

import pytest
from pytest_snapshot.plugin import Snapshot

from backend.blocks.dataforb2b._config import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_META_INPUT,
)
from backend.blocks.dataforb2b._enums import TypeaheadType
from backend.blocks.dataforb2b.typeahead import SearchFilterTypeaheadBlock


async def _run_and_capture(block: SearchFilterTypeaheadBlock, input_data):
    """Run the block with a mocked client, returning (outputs, call_args)."""
    outputs = {}
    with patch.object(
        SearchFilterTypeaheadBlock,
        "typeahead",
        new=AsyncMock(return_value={"results": [{"value": "Google"}]}),
    ) as mock_typeahead:
        async for name, value in block.run(input_data, credentials=TEST_CREDENTIALS):
            outputs[name] = value
        return outputs, mock_typeahead


@pytest.mark.asyncio
async def test_missing_query_raises_value_error():
    """Thread ED6e: missing 'q' should raise a clear ValueError, not an API 400."""
    block = SearchFilterTypeaheadBlock()
    input_data = SearchFilterTypeaheadBlock.Input(
        credentials=TEST_CREDENTIALS_META_INPUT,
        filter_type=TypeaheadType.COMPANY,
        q="",
        limit=5,
    )
    with pytest.raises(ValueError, match="'q'"):
        async for _ in block.run(input_data, credentials=TEST_CREDENTIALS):
            pass


@pytest.mark.asyncio
async def test_limit_clamped_to_upper_bound(snapshot: Snapshot):
    """Thread ED6e: limit above 20 must be clamped, not passed through raw."""
    block = SearchFilterTypeaheadBlock()
    input_data = SearchFilterTypeaheadBlock.Input(
        credentials=TEST_CREDENTIALS_META_INPUT,
        filter_type=TypeaheadType.COMPANY,
        q="google",
        limit=999,
    )
    outputs, mock_typeahead = await _run_and_capture(block, input_data)
    assert mock_typeahead.await_args.args[2] == 20
    assert outputs["values"] == ["Google"]
    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(
        json.dumps(outputs, indent=2, sort_keys=True),
        "typeahead_outputs",
    )


@pytest.mark.asyncio
async def test_limit_clamped_to_lower_bound():
    """Thread ED6e: limit of 0 or negative must be clamped up to at least 1."""
    block = SearchFilterTypeaheadBlock()
    input_data = SearchFilterTypeaheadBlock.Input(
        credentials=TEST_CREDENTIALS_META_INPUT,
        filter_type=TypeaheadType.COMPANY,
        q="google",
        limit=0,
    )
    _, mock_typeahead = await _run_and_capture(block, input_data)
    assert mock_typeahead.await_args.args[2] == 1


@pytest.mark.asyncio
async def test_type_is_validated_enum():
    """Thread ED7M/ED64: 'type' is a closed enum, invalid values are rejected."""
    with pytest.raises(ValueError):
        SearchFilterTypeaheadBlock.Input.model_validate(
            {
                "credentials": TEST_CREDENTIALS_META_INPUT,
                "filter_type": "not-a-real-type",
                "q": "google",
            }
        )
