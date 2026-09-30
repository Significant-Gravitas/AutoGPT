"""Regression tests for ReadSpreadsheetBlock (GitHub issue #14638).

The Excel branch of the block used to lose data silently:
  * literal "N/A"/"NULL"/"NaN"/"None" cells were coerced to empty strings
  * a single gap in an integer column upcast every value ("12" -> "12.0")
  * only the first worksheet of a workbook was ever read
  * ``skip_columns`` compared ``int`` indices against ``str`` entries, so it
    never skipped anything
  * the Excel-to-CSV round trip hardcoded a comma while the parse used the
    configured ``delimiter``

The workbook is built here rather than committed as a binary fixture: the
backend test suite has no binary fixtures, and building it makes the exact
difference between a literal "N/A" cell and a genuinely empty one explicit.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from openpyxl import Workbook

from backend.blocks.spreadsheet import ReadSpreadsheetBlock
from backend.data.execution import ExecutionContext
from backend.util.file import get_exec_file_path
from backend.util.type import MediaFileType

GRAPH_EXEC_ID = "spreadsheet-test-exec"

# (label, cell value) — None means a genuinely empty cell, not the string "None"
SHEET1_ROWS = [
    [1, "N/A", 12, "ok"],
    [2, "NULL", None, None],
    [3, "n/a", 7, "NaN"],
    [4, "None", 5, "nan"],
]


@pytest.fixture
def workbook_path(tmp_path: Path):
    """Write a two-sheet workbook into the block's exec_file directory."""
    workbook = Workbook()
    first = workbook.active
    assert first is not None
    first.title = "Sheet1"
    first.append(["id", "status", "qty", "note"])
    for row in SHEET1_ROWS:
        first.append(row)

    second = workbook.create_sheet("Q2")
    second.append(["id", "status"])
    second.append([9, "NA"])

    base = Path(get_exec_file_path(GRAPH_EXEC_ID, ""))
    base.mkdir(parents=True, exist_ok=True)
    target = base / "book.xlsx"
    workbook.save(target)
    yield target
    target.unlink(missing_ok=True)


async def read_rows(**inputs) -> list[dict[str, str]]:
    """Run the block once and return its ``rows`` output."""
    block = ReadSpreadsheetBlock()
    context = ExecutionContext(user_id="test-user", graph_exec_id=GRAPH_EXEC_ID)
    outputs = {}
    with patch("backend.util.file.scan_content_safe", return_value=None):
        async for name, value in block.run(
            block.input_schema(**inputs), execution_context=context
        ):
            outputs[name] = value
    return outputs["rows"]


@pytest.mark.asyncio
async def test_excel_keeps_literal_missing_value_strings(workbook_path):
    rows = await read_rows(file_input=MediaFileType("book.xlsx"))

    assert [row["status"] for row in rows] == ["N/A", "NULL", "n/a", "None"]
    assert [row["note"] for row in rows] == ["ok", "", "NaN", "nan"]


@pytest.mark.asyncio
async def test_excel_keeps_genuinely_empty_cells_empty(workbook_path):
    rows = await read_rows(file_input=MediaFileType("book.xlsx"))

    assert rows[1]["note"] == ""
    assert rows[1]["qty"] == ""


@pytest.mark.asyncio
async def test_excel_does_not_upcast_integers_to_floats(workbook_path):
    rows = await read_rows(file_input=MediaFileType("book.xlsx"))

    assert [row["qty"] for row in rows] == ["12", "", "7", "5"]
    assert [row["id"] for row in rows] == ["1", "2", "3", "4"]


@pytest.mark.asyncio
async def test_excel_defaults_to_the_first_sheet(workbook_path):
    rows = await read_rows(file_input=MediaFileType("book.xlsx"))

    assert len(rows) == len(SHEET1_ROWS)
    assert list(rows[0]) == ["id", "status", "qty", "note"]


@pytest.mark.asyncio
async def test_excel_reads_the_requested_sheet(workbook_path):
    rows = await read_rows(file_input=MediaFileType("book.xlsx"), sheet_name="Q2")

    assert rows == [{"id": "9", "status": "NA"}]


@pytest.mark.asyncio
async def test_excel_unknown_sheet_name_is_reported(workbook_path):
    with pytest.raises(ValueError, match="Q3"):
        await read_rows(file_input=MediaFileType("book.xlsx"), sheet_name="Q3")


@pytest.mark.asyncio
async def test_excel_honours_the_configured_delimiter(workbook_path):
    rows = await read_rows(file_input=MediaFileType("book.xlsx"), delimiter=";")

    assert list(rows[0]) == ["id", "status", "qty", "note"]
    assert rows[0]["status"] == "N/A"


@pytest.mark.asyncio
async def test_excel_cell_containing_the_delimiter_survives(tmp_path):
    workbook = Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.append(["name", "note"])
    sheet.append(["widget", "a;b"])
    base = Path(get_exec_file_path(GRAPH_EXEC_ID, ""))
    base.mkdir(parents=True, exist_ok=True)
    target = base / "semi.xlsx"
    workbook.save(target)
    try:
        rows = await read_rows(file_input=MediaFileType("semi.xlsx"), delimiter=";")
    finally:
        target.unlink(missing_ok=True)

    assert rows == [{"name": "widget", "note": "a;b"}]


@pytest.mark.asyncio
async def test_skip_columns_skips_by_positional_index():
    rows = await read_rows(contents="a,b,c\n1,2,3\n4,5,6", skip_columns=["0"])

    assert rows == [{"b": "2", "c": "3"}, {"b": "5", "c": "6"}]


@pytest.mark.asyncio
async def test_skip_columns_default_keeps_every_column():
    rows = await read_rows(contents="a,b,c\n1,2,3")

    assert rows == [{"a": "1", "b": "2", "c": "3"}]


@pytest.mark.asyncio
async def test_skip_columns_without_header_uses_the_same_indices():
    rows = await read_rows(contents="1,2,3", has_header=False, skip_columns=["0", "2"])

    assert rows == [{"1": "2"}]
