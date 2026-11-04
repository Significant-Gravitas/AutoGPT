import pytest

from backend.blocks.spreadsheet_test import read_rows, read_sheet_rows


@pytest.mark.asyncio
@pytest.mark.parametrize("header", [",,", " , , ", '"","",""'])
async def test_all_blank_csv_headers_preserve_every_data_row(header: str):
    rows = await read_rows(contents=f"{header}\nfirst,second,third\nlast,row,values")

    assert rows == [
        {"column_1": "first", "column_2": "second", "column_3": "third"},
        {"column_1": "last", "column_2": "row", "column_3": "values"},
    ]


@pytest.mark.asyncio
async def test_all_blank_excel_headers_preserve_every_data_row():
    rows = await read_sheet_rows(
        [[None, None, None], ["first", "second", "third"], ["last", "row", "values"]]
    )

    assert rows == [
        {"column_1": "first", "column_2": "second", "column_3": "third"},
        {"column_1": "last", "column_2": "row", "column_3": "values"},
    ]


@pytest.mark.asyncio
async def test_whitespace_headers_are_preserved_without_stripping():
    rows = await read_rows(contents=" , \na,b\nc,d", strip=False)

    assert rows == [{" ": "a", " _2": "b"}, {" ": "c", " _2": "d"}]
