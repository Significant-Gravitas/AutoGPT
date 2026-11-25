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

import datetime
import io
import zipfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from openpyxl import Workbook
from openpyxl.styles import Font

from backend.blocks import spreadsheet
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
    return (await read_outputs(**inputs))["rows"]


async def read_outputs(**inputs) -> dict[str, Any]:
    """Run the block once and return the last value of each output."""
    block = ReadSpreadsheetBlock()
    context = ExecutionContext(user_id="test-user", graph_exec_id=GRAPH_EXEC_ID)
    outputs = {}
    with patch("backend.util.file.scan_content_safe", return_value=None):
        async for name, value in block.run(
            block.input_schema(**inputs), execution_context=context
        ):
            outputs[name] = value
    return outputs


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
            "openpyxl.load_workbook",
            side_effect=AssertionError("the workbook was read"),
        ) as load_workbook,
        pytest.raises(ValueError, match="unpacks to more than"),
    ):
        await read_rows(file_input=MediaFileType("book.xlsx"))
    load_workbook.assert_not_called()


@pytest.mark.asyncio
async def test_excel_is_read_from_the_bytes_that_were_checked(workbook_path):
    import io

    import openpyxl

    from backend.blocks import spreadsheet
    from backend.util.file_content_parser import check_xlsx_unpacked_size

    checked: list[bytes] = []
    read: list[bytes] = []
    real_load_workbook = openpyxl.load_workbook
    original = workbook_path.read_bytes()

    def check_then_change_file(archive, **kwargs):
        checked.append(archive.getvalue())
        check_xlsx_unpacked_size(archive, **kwargs)
        workbook_path.write_bytes(b"not the workbook that was checked")

    def record_read(source, *args, **kwargs):
        assert isinstance(source, io.BytesIO)
        read.append(source.getvalue())
        return real_load_workbook(source, *args, **kwargs)

    with (
        patch.object(spreadsheet, "check_xlsx_unpacked_size", check_then_change_file),
        patch("openpyxl.load_workbook", record_read),
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
async def test_excel_sheet_listing_past_the_cells_cap_is_refused_unread():
    target = save_workbook("big.xlsx", {f"XFD{row}": None for row in range(1, 400)})
    try:
        with (
            patch.object(
                spreadsheet,
                "_rows_as_csv",
                side_effect=AssertionError("a row was read"),
            ) as rows_as_csv,
            pytest.raises(ValueError, match="lists more than"),
        ):
            await read_rows(file_input=MediaFileType(target.name))
    finally:
        target.unlink(missing_ok=True)
    rows_as_csv.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("far", ["XFD2", "XFE2"])
async def test_excel_row_wider_than_excel_allows_is_refused_unread(far):
    from openpyxl.worksheet._reader import WorkSheetParser

    target = save_workbook("wide.xlsx", {"A1": "id", far: 7})
    try:
        if far == "XFD2":
            rows = await read_rows(file_input=MediaFileType(target.name))
            assert [row["column_16384"] for row in rows] == ["7"]
            return
        with (
            patch.object(
                WorkSheetParser,
                "parse_row",
                side_effect=AssertionError("a row was read"),
            ) as parse_row,
            pytest.raises(ValueError, match="a row of more than 16,384 cells"),
        ):
            await read_rows(file_input=MediaFileType(target.name))
        parse_row.assert_not_called()
    finally:
        target.unlink(missing_ok=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sheet_name, refused", [("Big", True), ("Small", False), (None, False)]
)
async def test_excel_counts_the_sheet_it_reads(sheet_name, refused):
    workbook = Workbook()
    small = workbook.active
    assert small is not None
    small.title = "Small"
    small.append(["id"])
    small.append(["1"])
    big = workbook.create_sheet("Big")
    big["A1"], big["XFD400"] = "id", "far"
    base = Path(get_exec_file_path(GRAPH_EXEC_ID, ""))
    base.mkdir(parents=True, exist_ok=True)
    target = base / "two.xlsx"
    workbook.save(target)
    try:
        if refused:
            with pytest.raises(ValueError, match="too large for this block to output"):
                await read_rows(file_input=MediaFileType("two.xlsx"), sheet_name="Big")
        else:
            rows = await read_rows(
                file_input=MediaFileType("two.xlsx"), sheet_name=sheet_name
            )
            assert rows == [{"id": "1"}]
    finally:
        target.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_rows_past_the_names_cap_are_refused():
    contents = "h" * 100_000 + "\n" + "1\n" * 400

    with pytest.raises(ValueError, match="too large for this block to output"):
        await read_rows(contents=contents)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cap, at_cap, contents, skip_columns",
    [
        # 64 per row and per cell, plus UTF-8 bytes: "é" counts 2.
        ("MAX_ROWS_SIZE", 393, "a,b\né,x\ny,z\n", []),
        ("MAX_ROWS_SIZE", 128, "a,b\n1,2\n3,4\n", ["0", "1"]),
        ("MAX_ROWS_NAMES_SIZE", 8, "ab,cd\n1,2\n3,4\n", []),
    ],
    ids=["bytes-and-items", "skipped-columns", "names"],
)  # fmt: skip
@pytest.mark.parametrize("past_cap", [False, True])
async def test_rows_at_a_cap_are_read_and_one_byte_past_it_refused(
    monkeypatch, cap, at_cap, contents, skip_columns, past_cap
):
    monkeypatch.setattr(spreadsheet, cap, at_cap - past_cap)

    if past_cap:
        with pytest.raises(ValueError, match="too large for this block to output"):
            await read_rows(contents=contents, skip_columns=skip_columns)
    else:
        assert len(await read_rows(contents=contents, skip_columns=skip_columns)) == 2


def save_workbook(name: str, cells: dict[str, object]) -> Path:
    """Write a one-sheet workbook into the block's exec_file directory.

    A ``None`` cell is written as an empty bold cell, listed in the file.
    """
    workbook = Workbook()
    sheet = workbook.active
    assert sheet is not None
    for ref, value in cells.items():
        sheet[ref] = value
        if value is None:
            sheet[ref].font = Font(bold=True)
    base = Path(get_exec_file_path(GRAPH_EXEC_ID, ""))
    base.mkdir(parents=True, exist_ok=True)
    target = base / name
    workbook.save(target)
    return target


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


# Blank lines before the header and between rows count toward skip_rows.
PAGED_CSV = "\n\nid,name\n" + "".join(
    f"{n},n{n}\n" + ("\n" if n % 5 == 0 else "") for n in range(1, 24)
)


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["csv", "excel"])
@pytest.mark.parametrize("max_rows", [1, 4, 7, 100])
async def test_reading_in_pages_gives_every_row_once(source, max_rows):
    if source == "excel":
        cells: dict[str, object] = {"A1": "id", "B1": "name"}
        for n in range(2, 25):
            if n != 9:  # an empty row inside the data
                cells[f"A{n}"], cells[f"B{n}"] = n, f"n{n}"
        target = save_workbook("pages.xlsx", cells)
        inputs: dict[str, Any] = {"file_input": MediaFileType(target.name)}
    else:
        inputs = {"contents": PAGED_CSV}
    whole = await read_rows(**inputs)

    pages = await read_in_pages(max_rows, **inputs)

    assert len(whole) == 23
    assert all(1 <= len(page) <= max_rows for page in pages)
    assert [row for page in pages for row in page] == whole


@pytest.mark.asyncio
async def test_a_read_too_large_to_output_at_once_is_refused_unless_paged(
    monkeypatch,
):
    contents = "a,b\n" + "".join(f"{n},x\n" for n in range(10))
    whole = await read_rows(contents=contents)
    # 64 per row and per cell, plus "a"/"b" and the two values: 196 a row.
    monkeypatch.setattr(spreadsheet, "MAX_ROWS_SIZE", 3 * 196)

    with pytest.raises(ValueError, match="Read it in pages: set max_rows"):
        await read_outputs(contents=contents)

    pages = await read_in_pages(100, contents=contents)
    assert [len(page) for page in pages] == [3, 3, 3, 1]
    assert [row for page in pages for row in page] == whole


@pytest.mark.asyncio
async def test_excel_cells_holding_the_escapechar_read_as_written():
    folders = ["C:\\temp\\", 'say \\"hi\\"', "D:\\data\\", "plain"]
    cells: dict[str, object] = {"A1": "name", "B1": "folder"}
    for n, folder in enumerate(folders, start=2):
        cells[f"A{n}"], cells[f"B{n}"] = f"r{n}", folder
    inputs = {"file_input": MediaFileType(save_workbook("folders.xlsx", cells).name)}

    whole = await read_rows(**inputs)
    pages = await read_in_pages(1, **inputs)

    assert [row["folder"] for row in whole] == folders
    assert [row for page in pages for row in page] == whole


async def read_in_pages(max_rows: int, **inputs) -> list[list[dict[str, str]]]:
    """Every page of a read, each starting at the one before's next_skip_rows."""
    pages, skip_rows = [], 0
    while len(pages) < 30:
        page = await read_outputs(**inputs, max_rows=max_rows, skip_rows=skip_rows)
        pages.append(page["rows"])
        if "next_skip_rows" not in page:
            return pages
        skip_rows = page["next_skip_rows"]
    raise AssertionError(f"30 pages and still not done, at skip_rows={skip_rows}")


@pytest.mark.asyncio
@pytest.mark.parametrize("delimiter, quotechar", [(",", '"'), (";", "'")])
async def test_excel_reads_as_read_excel_does(monkeypatch, delimiter, quotechar):
    """Read Spreadsheet streams a sheet, and must give the rows pandas'
    read_excel gave: whole numbers, dates, booleans, quoting, padding."""
    # One row per chunk, so every value crosses a chunk boundary.
    monkeypatch.setattr(spreadsheet, "XLSX_CHUNK_CELLS", 1)
    columns = [
        ["flag", False, 0, True, 1, 0, None, True],
        ["num", 12, 1.5, 3.0, -0.0, 1e20, None, 0.1],
        ["when", datetime.datetime(2024, 1, 2, 3, 4), datetime.date(2023, 12, 31)],
        ["text", "a;b,c", 'say "hi"', "it's", "line\nbreak", " pad ", "N/A", "007"],
        [None, None, None, "past the header"],
    ]
    cells: dict[str, object] = {}
    for col, values in zip("ABCDE", columns):
        for row, value in enumerate(values, start=1):
            if value is not None:
                cells[f"{col}{row}"] = value
    target = save_workbook("mixed.xlsx", cells)
    dialect = {"delimiter": delimiter, "quotechar": quotechar}
    expected = await read_rows(contents=_read_excel_csv(target, **dialect), **dialect)
    assert expected[0]["flag"] == "False" and expected[1]["flag"] == "False"

    whole = await read_rows(file_input=MediaFileType(target.name), **dialect)
    page = await read_rows(
        file_input=MediaFileType(target.name), skip_rows=2, max_rows=3, **dialect
    )

    assert whole == expected
    assert page == expected[2:5]


def _read_excel_csv(target: Path, delimiter: str, quotechar: str) -> str:
    """The CSV the block read an Excel sheet as before it streamed sheets."""
    import pandas as pd

    frame = pd.read_excel(
        io.BytesIO(target.read_bytes()),
        header=None,
        dtype=object,
        keep_default_na=False,
    )
    text = io.StringIO()
    frame.to_csv(text, index=False, header=False, sep=delimiter, quotechar=quotechar)
    return text.getvalue()
