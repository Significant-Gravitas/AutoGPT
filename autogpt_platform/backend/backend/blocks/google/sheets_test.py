"""Tests for Google Sheets blocks — edge cases around credentials handling."""

import pytest

from backend.blocks.google.sheets import GoogleSheetsReadBlock
from backend.util.exceptions import BlockExecutionError


async def test_sheets_read_no_spreadsheet_yields_clean_error():
    """Executing GoogleSheetsReadBlock with no spreadsheet/credentials should
    raise a clean BlockExecutionError, not a BlockUnknownError wrapping a
    TypeError about a missing kwarg."""
    block = GoogleSheetsReadBlock()
    input_data = {"range": "Sheet1!A1:B2"}  # no spreadsheet, no credentials

    with pytest.raises(BlockExecutionError, match="No spreadsheet selected"):
        async for _ in block.execute(input_data):
            pass
