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

import zipfile
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


# The .xlsx caps: 50 MiB per member, 200 MiB per workbook.
XLSX_MEMBER_CAP = 50 * 1024 * 1024


def add_member(workbook: Path, name: str, size: int) -> None:
    """Append a member of ``size`` zero bytes, written in pieces."""
    with (
        zipfile.ZipFile(workbook, "a", zipfile.ZIP_DEFLATED) as archive,
        archive.open(name, "w") as member,
    ):
        for start in range(0, size, 4 * 1024 * 1024):
            member.write(b"\0" * min(4 * 1024 * 1024, size - start))


@pytest.mark.asyncio
async def test_excel_member_unpacking_past_the_cap_is_refused_unread(workbook_path):
    add_member(workbook_path, "xl/media/image1.png", XLSX_MEMBER_CAP + 1)
    assert workbook_path.stat().st_size < 1_000_000

    with (
        patch(
            "pandas.read_excel",
            side_effect=AssertionError("the workbook was read"),
        ) as read_excel,
        pytest.raises(ValueError, match="unpacks to more than"),
    ):
        await read_rows(file_input=MediaFileType("book.xlsx"))
    read_excel.assert_not_called()


@pytest.mark.asyncio
async def test_excel_is_read_from_the_bytes_that_were_checked(workbook_path):
    import io

    import pandas as pd

    from backend.blocks import spreadsheet
    from backend.util.file_content_parser import check_xlsx_unpacked_size

    checked: list[bytes] = []
    read: list[bytes] = []
    real_read_excel = pd.read_excel
    original = workbook_path.read_bytes()

    def check_then_change_file(archive, **kwargs):
        checked.append(archive.getvalue())
        check_xlsx_unpacked_size(archive, **kwargs)
        workbook_path.write_bytes(b"not the workbook that was checked")

    def record_read(source, *args, **kwargs):
        assert isinstance(source, io.BytesIO)
        read.append(source.getvalue())
        return real_read_excel(source, *args, **kwargs)

    with (
        patch.object(spreadsheet, "check_xlsx_unpacked_size", check_then_change_file),
        patch("pandas.read_excel", record_read),
    ):
        rows = await read_rows(file_input=MediaFileType("book.xlsx"))

    assert checked == [original]
    assert read == checked
    assert rows[0]["status"] == "N/A"


@pytest.mark.asyncio
async def test_excel_member_at_the_cap_is_read(workbook_path):
    add_member(workbook_path, "xl/media/image1.png", XLSX_MEMBER_CAP)

    rows = await read_rows(file_input=MediaFileType("book.xlsx"))

    assert rows[0]["status"] == "N/A"


@pytest.mark.asyncio
async def test_excel_member_past_the_cap_is_reported_once(workbook_path):
    add_member(workbook_path, "xl/media/image1.png", XLSX_MEMBER_CAP + 1)

    with (
        patch("backend.util.file_content_parser.sentry_sdk.capture_message") as report,
        pytest.raises(ValueError, match="unpacks to more than"),
    ):
        await read_rows(file_input=MediaFileType("book.xlsx"))

    report.assert_called_once()
    assert report.call_args.args == ("xlsx workbook over unpack limit",)
    assert report.call_args.kwargs["level"] == "warning"
    assert report.call_args.kwargs["tags"] == {
        "xlsx_limit": "MAX_XLSX_MEMBER_BYTES",
        "xlsx_source": "spreadsheet_block",
    }


@pytest.mark.asyncio
async def test_excel_within_the_caps_is_not_reported(workbook_path):
    with patch("backend.util.file_content_parser.sentry_sdk.capture_message") as report:
        await read_rows(file_input=MediaFileType("book.xlsx"))

    report.assert_not_called()


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
