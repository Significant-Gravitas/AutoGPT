"""Tests for the SpecStream List Rows block."""

from unittest.mock import AsyncMock

import pytest

from backend.blocks.specstream._test_helpers import run_block
from backend.blocks.specstream._testdata import TEST_FIND_ID, TEST_ROWS_PAGE
from backend.blocks.specstream.list_rows import SpecStreamListRowsBlock


@pytest.mark.asyncio
async def test_list_rows_asks_only_for_the_rows_still_needed_and_keeps_each_source():
    first = TEST_ROWS_PAGE.model_copy(update={"next_cursor": "cur_2"})
    last = TEST_ROWS_PAGE.model_copy(update={"next_cursor": None})
    block = SpecStreamListRowsBlock()
    block.read_page = AsyncMock(side_effect=[first, last])

    outputs = await run_block(block, run_id=TEST_FIND_ID, max_rows=2)

    assert [call.args[2:] for call in block.read_page.await_args_list] == [
        (None, 2),
        ("cur_2", 1),
    ]
    assert outputs["row_count"] == 2
    assert outputs["next_cursor"] == ""
    cell = outputs["rows"][0].cells[0]
    assert (cell.source_url, cell.read_at) == (
        "https://acme.example/about",
        "2026-11-03T09:30:00+00:00",
    )
    assert outputs["rows"][0].values["website"] == "https://acme.example"
    sources = {
        cell.column_id: (cell.source_url, cell.read_at)
        for cell in outputs["rows"][0].cells
    }
    witness = ("https://acme.example/team", "2026-11-03T09:31:00+00:00")
    assert sources["detail:w1"] == witness
    assert sources["matches"] == witness
