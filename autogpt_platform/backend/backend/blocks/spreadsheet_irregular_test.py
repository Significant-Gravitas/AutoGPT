"""Regression tests for ReadSpreadsheetBlock on irregular input.

#15269: a row longer than the header raised IndexError, and skip_rows past the
end raised "async generator raised StopIteration".
#15270: a repeated header overwrote the earlier column, a blank header cell
became the key "", and a blank line became an empty row.

Excel sheets go through the same header rules as CSV, so pandas' own names
("a.1", "Unnamed: 2") never reach the output.
"""

from pathlib import Path

import pytest
from openpyxl import Workbook

from backend.blocks.spreadsheet_test import GRAPH_EXEC_ID, read_rows
from backend.util.file import get_exec_file_path
from backend.util.type import MediaFileType


async def read_sheet_rows(rows: list[list[object]], **inputs) -> list[dict[str, str]]:
    """Write ``rows`` to a one-sheet workbook (None = no cell) and read it."""
    workbook = Workbook()
    sheet = workbook.active
    assert sheet is not None
    for r, row in enumerate(rows, start=1):
        for c, value in enumerate(row, start=1):
            if value is not None:
                sheet.cell(r, c, value)
    base = Path(get_exec_file_path(GRAPH_EXEC_ID, ""))
    base.mkdir(parents=True, exist_ok=True)
    target = base / "irregular.xlsx"
    workbook.save(target)
    try:
        return await read_rows(file_input=MediaFileType("irregular.xlsx"), **inputs)
    finally:
        target.unlink(missing_ok=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "contents, skip_rows",
    [
        pytest.param("a,b\n1,2", 5, id="skip-past-the-end"),
        pytest.param("a,b\n1,2", 1, id="skip-to-the-end"),
        pytest.param("a,b", 0, id="header-only"),
        pytest.param("\n", 0, id="blank-line-only"),
    ],
)
async def test_input_without_data_rows_yields_no_rows(contents: str, skip_rows: int):
    assert await read_rows(contents=contents, skip_rows=skip_rows) == []


@pytest.mark.asyncio
async def test_empty_contents_is_rejected():
    with pytest.raises(ValueError, match="contents"):
        await read_rows(contents="")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "contents, expected",
    [
        pytest.param(
            "a,b\n1,2,3\n4,5",
            [{"a": "1", "b": "2", "column_3": "3"}, {"a": "4", "b": "5"}],
            id="field-past-the-header",
        ),
        pytest.param(
            "a\n1,2\n3,4",
            [{"a": "1", "column_2": "2"}, {"a": "3", "column_2": "4"}],
            id="field-past-the-header-same-key-every-row",
        ),
        pytest.param(
            "a,column_3\n1,2,3",
            [{"a": "1", "column_3": "2", "column_3_2": "3"}],
            id="field-past-the-header-name-taken",
        ),
        pytest.param(
            "a,a,,a\n1,2,3,4",
            [{"a": "1", "a_2": "2", "column_3": "3", "a_3": "4"}],
            id="repeated-and-blank-headers",
        ),
        pytest.param(
            "a,a_2,a\n1,2,3",
            [{"a": "1", "a_2": "2", "a_3": "3"}],
            id="repeat-after-a-suffixed-name",
        ),
        pytest.param(
            "a,a,a_2\n1,2,3",
            [{"a": "1", "a_3": "2", "a_2": "3"}],
            id="repeat-before-a-suffixed-name",
        ),
        pytest.param(
            ",column_1\n1,2",
            [{"column_1_2": "1", "column_1": "2"}],
            id="blank-before-a-fallback-name",
        ),
    ],
)
async def test_every_value_keeps_its_own_key(contents: str, expected: list[dict]):
    assert await read_rows(contents=contents) == expected


@pytest.mark.asyncio
async def test_blank_lines_are_not_rows():
    rows = await read_rows(contents="a,b\n1,2\n\n3,4\n")

    assert rows == [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]


@pytest.mark.asyncio
async def test_empty_field_record_is_kept():
    rows = await read_rows(contents="a,b\n1,2\n,\n3,4")

    assert rows == [{"a": "1", "b": "2"}, {"a": "", "b": ""}, {"a": "3", "b": "4"}]


@pytest.mark.asyncio
async def test_leading_blank_line_does_not_steal_the_header():
    rows = await read_rows(contents="\na,b\n1,2")

    assert rows == [{"a": "1", "b": "2"}]


@pytest.mark.asyncio
async def test_leading_blank_line_counts_toward_skip_rows():
    rows = await read_rows(contents="\na,b\n1,2\n3,4", skip_rows=1)

    assert rows == [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("header", [",,", " , , ", '"","",""'])
async def test_all_blank_header_keeps_every_data_row(header: str):
    rows = await read_rows(contents=f"{header}\nfirst,second,third\nlast,row,values")

    assert rows == [
        {"column_1": "first", "column_2": "second", "column_3": "third"},
        {"column_1": "last", "column_2": "row", "column_3": "values"},
    ]


@pytest.mark.asyncio
async def test_whitespace_headers_are_kept_without_stripping():
    rows = await read_rows(contents=" , \na,b\nc,d", strip=False)

    assert rows == [{" ": "a", " _2": "b"}, {" ": "c", " _2": "d"}]


# --- Excel goes through the same header handling as CSV --------------------


@pytest.mark.asyncio
async def test_excel_repeated_and_blank_headers_match_csv():
    rows = await read_sheet_rows([["a", "a", None, "b"], [1, 2, 3, 4]])

    assert rows == await read_rows(contents="a,a,,b\n1,2,3,4")
    assert rows == [{"a": "1", "a_2": "2", "column_3": "3", "b": "4"}]


@pytest.mark.asyncio
async def test_excel_repeat_after_a_suffixed_name_matches_csv():
    rows = await read_sheet_rows([["a", "a_2", "a"], [1, 2, 3]])

    assert rows == await read_rows(contents="a,a_2,a\n1,2,3")
    assert rows == [{"a": "1", "a_2": "2", "a_3": "3"}]


@pytest.mark.asyncio
async def test_excel_without_header_keeps_first_row_values():
    rows = await read_sheet_rows([["a", "a", None], [1, 2, 3]], has_header=False)

    assert rows == [{"0": "a", "1": "a", "2": ""}, {"0": "1", "1": "2", "2": "3"}]


@pytest.mark.asyncio
async def test_excel_all_blank_header_keeps_every_data_row():
    rows = await read_sheet_rows(
        [[None, None, None], ["first", "second", "third"], ["last", "row", "values"]]
    )

    assert rows == [
        {"column_1": "first", "column_2": "second", "column_3": "third"},
        {"column_1": "last", "column_2": "row", "column_3": "values"},
    ]


@pytest.mark.asyncio
async def test_excel_empty_first_row_keeps_data_and_empty_records():
    rows = await read_sheet_rows([[], ["a", "b"], [1, 2], [None, None], [3, 4]])

    assert rows == [
        {"column_1": "a", "column_2": "b"},
        {"column_1": "1", "column_2": "2"},
        {"column_1": "", "column_2": ""},
        {"column_1": "3", "column_2": "4"},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("skip_rows", [0, 5])
async def test_excel_header_only_sheet_yields_no_rows(skip_rows: int):
    assert await read_sheet_rows([["a", "b"]], skip_rows=skip_rows) == []
