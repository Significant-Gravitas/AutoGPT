import asyncio
import csv
import io
from collections.abc import Callable, Generator, Iterable, Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.execution import ExecutionContext
from backend.data.model import ContributorDetails, SchemaField
from backend.util.file import get_exec_file_path, store_media_file
from backend.util.file_content_parser import check_xlsx_unpacked_size, xlsx_sheet_shape
from backend.util.type import MediaFileType

# What the rows output may hold: the UTF-8 bytes of its names and values plus
# ROWS_ITEM_SIZE per row and per cell, and the UTF-8 bytes of the names it holds.
MAX_ROWS_SIZE = 100_000_000
ROWS_ITEM_SIZE = 64
MAX_ROWS_NAMES_SIZE = 10_000_000
# With max_rows empty, a read past those limits is refused rather than cut to its
# first page, so a workflow never takes part of a spreadsheet for all of it.
PAGE_WITHOUT_MAX_ROWS = False
# Rows of an Excel sheet are converted a few at a time, up to either budget.
XLSX_CHUNK_CELLS = 50_000
XLSX_CHUNK_CHARACTERS = 1_000_000


class ReadSpreadsheetBlock(Block):
    class Input(BlockSchemaInput):
        contents: str | None = SchemaField(
            description="The contents of the CSV/spreadsheet data to read",
            placeholder="a, b, c\n1,2,3\n4,5,6",
            default=None,
            advanced=False,
        )
        file_input: MediaFileType | None = SchemaField(
            description="CSV or Excel file to read from (URL, data URI, or local path). Excel files are automatically converted to CSV",
            default=None,
            advanced=False,
        )
        sheet_name: str | None = SchemaField(
            description="Name of the worksheet to read from an Excel file. Defaults to the first sheet.",
            default=None,
        )
        delimiter: str = SchemaField(
            description="The delimiter used in the CSV/spreadsheet data",
            default=",",
        )
        quotechar: str = SchemaField(
            description="The character used to quote fields",
            default='"',
        )
        escapechar: str = SchemaField(
            description="The character used to escape the delimiter",
            default="\\",
        )
        has_header: bool = SchemaField(
            description="Whether the CSV file has a header row",
            default=True,
        )
        skip_rows: int = SchemaField(
            description="The number of rows to skip from the start of the file",
            default=0,
        )
        strip: bool = SchemaField(
            description="Whether to strip whitespace from the values",
            default=True,
        )
        skip_columns: list[str] = SchemaField(
            description="The columns to skip from the start of the row",
            default_factory=list,
        )
        produce_singular_result: bool = SchemaField(
            description="If True, yield individual 'row' outputs only (can be slow). If False, yield both 'rows' (all data)",
            default=False,
        )
        max_rows: int | None = SchemaField(
            description="The most rows to read after skip_rows. Set it to read a large spreadsheet in pages, starting each page at next_skip_rows; empty reads every row, or fails if they are too large to output at once.",
            default=None,
            ge=1,
        )

    class Output(BlockSchemaOutput):
        row: dict[str, str] = SchemaField(
            description="The data produced from each row in the spreadsheet"
        )
        rows: list[dict[str, str]] = SchemaField(
            description="All the data in the spreadsheet as a list of rows"
        )
        next_skip_rows: int = SchemaField(
            description="Only when rows are left after this read: the skip_rows that reads the next page"
        )

    def __init__(self):
        super().__init__(
            id="acf7625e-d2cb-4941-bfeb-2819fc6fc015",
            input_schema=ReadSpreadsheetBlock.Input,
            output_schema=ReadSpreadsheetBlock.Output,
            description="Reads CSV and Excel files and outputs the data as a list of dictionaries and individual rows. Excel files are automatically converted to CSV format.",
            contributors=[ContributorDetails(name="Nicholas Tindle")],
            categories={BlockCategory.TEXT, BlockCategory.DATA},
            test_input=[
                {
                    "contents": "a, b, c\n1,2,3\n4,5,6",
                    "produce_singular_result": False,
                },
                {
                    "contents": "a, b, c\n1,2,3\n4,5,6",
                    "produce_singular_result": True,
                },
            ],
            test_output=[
                (
                    "rows",
                    [
                        {"a": "1", "b": "2", "c": "3"},
                        {"a": "4", "b": "5", "c": "6"},
                    ],
                ),
                ("row", {"a": "1", "b": "2", "c": "3"}),
                ("row", {"a": "4", "b": "5", "c": "6"}),
            ],
            effect=BlockEffect.READ,
        )

    async def run(
        self, input_data: Input, *, execution_context: ExecutionContext, **_kwargs
    ) -> BlockOutput:
        # Determine data source - prefer file_input if provided, otherwise use contents
        file_path = None
        if input_data.file_input:
            stored_file_path = await store_media_file(
                file=input_data.file_input,
                execution_context=execution_context,
                return_format="for_local_processing",
            )

            # Get full file path
            assert execution_context.graph_exec_id  # Validated by store_media_file
            file_path = get_exec_file_path(
                execution_context.graph_exec_id, stored_file_path
            )
            if not Path(file_path).exists():
                raise ValueError(f"File does not exist: {file_path}")
        elif not input_data.contents:
            raise ValueError("Either 'contents' or 'file_input' must be provided")

        rows, next_skip_rows = await asyncio.to_thread(
            _read_rows, input_data, file_path
        )

        if input_data.produce_singular_result:
            for processed_row in rows:
                yield "row", processed_row
        else:
            yield "rows", rows
        if next_skip_rows is not None:
            yield "next_skip_rows", next_skip_rows


def _read_rows(
    input_data: ReadSpreadsheetBlock.Input, file_path: str | None
) -> tuple[list[dict[str, str]], int | None]:
    """The rows to output, and the skip_rows of the next page when rows are left."""
    with _csv_lines(input_data, file_path) as (lines, skip_rows):
        reader = csv.reader(
            lines,
            delimiter=input_data.delimiter,
            quotechar=input_data.quotechar,
            escapechar=input_data.escapechar,
        )

        # Blank lines before the header count toward skip_rows. A header whose
        # cells are all empty is still the header: skipping it would make the
        # first data row the header and lose its values.
        blank_lines = 0
        header = None
        if input_data.has_header:
            # Empty input has no header row.
            header = next(reader, None)
            while header == []:
                blank_lines += 1
                skip_rows = max(skip_rows - 1, 0)
                header = next(reader, None)
            if header is not None:
                if input_data.strip:
                    header = [h.strip() for h in header]
                header = _unique_column_names(header)

        for _ in range(skip_rows):
            if next(reader, None) is None:
                break

        header_names = set(header or [])
        extra_names: dict[int, str] = {}

        def column_key(i: int) -> str:
            if not header:
                return str(i)
            if i < len(header):
                return header[i]
            # A row with more fields than the header keeps the extra values,
            # under the same key in every row.
            if i not in extra_names:
                extra_names[i] = _claim_name(f"column_{i + 1}", header_names, {})
            return extra_names[i]

        def process_row(row):
            data = {}
            for i, value in enumerate(row):
                if str(i) not in input_data.skip_columns:
                    data[column_key(i)] = value.strip() if input_data.strip else value
            return data

        rows, read = _collect_rows(reader, process_row, input_data.max_rows)
    if read is None:
        return rows, None
    return rows, max(input_data.skip_rows, blank_lines) + read


@contextmanager
def _csv_lines(
    input_data: ReadSpreadsheetBlock.Input, file_path: str | None
) -> Iterator[tuple[Iterable[str], int]]:
    """The lines to parse as CSV, and how many rows after the header to skip."""
    if file_path is None:
        yield io.StringIO(input_data.contents or ""), input_data.skip_rows
    elif Path(file_path).suffix.lower() in [".xlsx", ".xls"]:
        # Excel files are converted to CSV, skipping rows before converting them.
        with closing(_xlsx_csv_lines(file_path, input_data)) as lines:
            yield lines, 0
    else:
        with open(file_path, encoding="utf-8") as lines:
            yield lines, input_data.skip_rows


def _xlsx_csv_lines(
    file_path: str, input_data: ReadSpreadsheetBlock.Input
) -> Generator[str, None, None]:
    """The sheet's CSV lines, as pandas' ``read_excel`` then ``to_csv`` write them,
    from the header row on and past the rows skip_rows skips, a few rows at a time.
    """
    try:
        from openpyxl import load_workbook

        workbook_file = io.BytesIO(Path(file_path).read_bytes())
        check_xlsx_unpacked_size(workbook_file, source="spreadsheet_block")
        workbook = load_workbook(
            workbook_file, read_only=True, data_only=True, keep_links=False
        )
        try:
            sheet = _worksheet(workbook, input_data.sheet_name)
            last_row, width = xlsx_sheet_shape(sheet, source="spreadsheet_block")
            skipped = range(
                int(input_data.has_header) + 1,
                int(input_data.has_header) + input_data.skip_rows + 1,
            )
            per_chunk = max(1, XLSX_CHUNK_CELLS // max(width, 1))
            first_seen: dict[tuple[int, bool], Any] = {}
            chunk: list[list[Any]] = []
            characters = 0
            for number, cells in enumerate(sheet.iter_rows(max_row=last_row), 1):
                # A skipped row is read too: its values set what later ones print as.
                values = _row_values(cells, width, first_seen)
                if number in skipped:
                    continue
                row_characters = sum(len(str(value)) for value in values)
                if row_characters > MAX_ROWS_SIZE:
                    raise ValueError(
                        "a row of the sheet is too large for this block to read"
                    )
                chunk.append(values)
                characters += row_characters
                if len(chunk) == per_chunk or characters >= XLSX_CHUNK_CHARACTERS:
                    yield from _rows_as_csv(chunk, input_data)
                    chunk, characters = [], 0
            if chunk:
                yield from _rows_as_csv(chunk, input_data)
        finally:
            workbook.close()
    except Exception as e:
        raise ValueError(f"Unable to read Excel file: {e}") from e


def _worksheet(workbook: Any, sheet_name: str | None) -> Any:
    # The sheet read_excel picks, with its error for a name the workbook lacks.
    if not sheet_name:
        return workbook.worksheets[0]
    if sheet_name not in [sheet.title for sheet in workbook.worksheets]:
        raise ValueError(f"Worksheet named '{sheet_name}' not found")
    return workbook[sheet_name]


def _row_values(
    cells: tuple[Any, ...], width: int, first_seen: dict[tuple[int, bool], Any]
) -> list[Any]:
    """A row's values as read_excel trims and pads them.

    pandas' parser gives each column's later values that equal an earlier one
    the earlier one, so a 0 below a False reads "False". Among cell values only
    0/False and 1/True are equal and print differently, so ``first_seen`` keeps
    those per column, for the whole sheet.
    """
    values = [_cell_value(cell) for cell in cells]
    while values and values[-1] == "":
        values.pop()
    for column, value in enumerate(values):
        if isinstance(value, int) and value in (0, 1):
            values[column] = first_seen.setdefault((column, value == 1), value)
    return values + [""] * (width - len(values))


def _rows_as_csv(
    data: list[list[Any]], input_data: ReadSpreadsheetBlock.Input
) -> io.StringIO:
    """Rows of values as read_excel parses them, written to CSV."""
    from pandas.io.parsers import TextParser

    # dtype=object and keep_default_na=False keep the sheet's own values:
    # "N/A"/"NULL"/"NaN" stay text instead of being read as missing, and a gap
    # in a numeric column doesn't upcast it to float (turning 12 into "12.0").
    # header=None hands every row, header included, to the CSV reader, so
    # pandas doesn't rename duplicate or blank headers.
    frame = TextParser(
        data,
        header=None,
        dtype=object,
        keep_default_na=False,
        skip_blank_lines=False,
    ).read()
    # Write the CSV with the same dialect it is parsed with, otherwise a
    # non-default delimiter yields a single column.
    text = io.StringIO()
    frame.to_csv(
        text,
        index=False,
        header=False,
        sep=input_data.delimiter,
        quotechar=input_data.quotechar,
    )
    text.seek(0)
    return text


def _cell_value(cell: Any) -> Any:
    # pandas' openpyxl conversion: whole numbers as int, errors ("e") as missing.
    if cell.value is None:
        return ""
    if cell.data_type == "e":
        return float("nan")
    if cell.data_type == "n":
        whole = int(cell.value)
        return whole if whole == cell.value else float(cell.value)
    return cell.value


def _collect_rows(
    reader: Iterable[list[str]],
    process_row: Callable[[list[str]], dict[str, str]],
    max_rows: int | None,
) -> tuple[list[dict[str, str]], int | None]:
    """Process up to ``max_rows`` rows within the rows limits, and how many rows
    the reader gave for them, blank ones included, when rows are left after them.
    """
    rows = []
    read = size = names_size = 0
    for row in reader:
        if row and len(rows) == max_rows:
            return rows, read
        read += 1
        # csv.reader yields [] for a blank line; don't turn those into {} rows.
        if not row:
            continue
        processed = process_row(row)
        names = sum(_utf8_size(name) for name in processed)
        values = sum(_utf8_size(value) for value in processed.values())
        names_size += names
        size += ROWS_ITEM_SIZE * (1 + len(processed)) + names + values
        if size > MAX_ROWS_SIZE or names_size > MAX_ROWS_NAMES_SIZE:
            if not rows:
                raise ValueError(
                    "A row of the spreadsheet is too large for this block to "
                    "output; skip some of its columns."
                )
            if max_rows is None and not PAGE_WITHOUT_MAX_ROWS:
                raise ValueError(
                    "The spreadsheet is too large for this block to output at "
                    "once. Read it in pages: set max_rows, and start each page "
                    "at the next_skip_rows of the one before. Or skip some columns."
                )
            return rows, read - 1
        rows.append(processed)
    return rows, None


def _utf8_size(text: str) -> int:
    return len(text) if text.isascii() else len(text.encode())


def _unique_column_names(header: list[str]) -> list[str]:
    """Turn header cells into distinct dict keys.

    The first cell with a given text keeps it. A blank cell becomes
    ``column_<position>`` and a repeat gets ``_2``, ``_3``…, skipping every
    name a header cell has in the file, so no column overwrites another and a
    key always names the column whose header it is.
    """
    taken = {name for name in header if name}
    suffixes: dict[str, int] = {}
    seen: set[str] = set()
    names = []
    for i, name in enumerate(header):
        if name and name not in seen:
            seen.add(name)
            names.append(name)
        else:
            names.append(_claim_name(name or f"column_{i + 1}", taken, suffixes))
    return names


def _claim_name(base: str, taken: set[str], suffixes: dict[str, int]) -> str:
    """Return ``base``, or ``base_<n>`` with the lowest free n, and mark it taken.

    ``suffixes`` remembers where each base's search stopped. Names are only
    ever added to ``taken``, so every lower suffix is still taken, and a
    header of many repeats is named in linear time.
    """
    name, n = base, suffixes.get(base, 2)
    while name in taken:
        name, n = f"{base}_{n}", n + 1
    suffixes[base] = n
    taken.add(name)
    return name
