import asyncio
import io
from collections.abc import Callable, Iterable
from pathlib import Path

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
from backend.util.file import MAX_FILE_SIZE_BYTES, get_exec_file_path, store_media_file
from backend.util.file_content_parser import (
    check_xlsx_sheet_size,
    check_xlsx_unpacked_size,
)
from backend.util.type import MediaFileType

# What the rows output may hold: the UTF-8 bytes of its names and values plus
# ROWS_ITEM_SIZE per row and per cell, and the UTF-8 bytes of the names it holds.
MAX_ROWS_SIZE = 100_000_000
ROWS_ITEM_SIZE = 64
MAX_ROWS_NAMES_SIZE = 10_000_000


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

    class Output(BlockSchemaOutput):
        row: dict[str, str] = SchemaField(
            description="The data produced from each row in the spreadsheet"
        )
        rows: list[dict[str, str]] = SchemaField(
            description="All the data in the spreadsheet as a list of rows"
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
        import csv
        from io import StringIO

        # Determine data source - prefer file_input if provided, otherwise use contents
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

            # Check if file is an Excel file and convert to CSV
            file_extension = Path(file_path).suffix.lower()

            if file_extension in [".xlsx", ".xls"]:
                # Handle Excel files
                try:
                    import pandas as pd

                    def _read_xlsx() -> str:
                        workbook = Path(file_path).read_bytes()
                        check_xlsx_unpacked_size(
                            io.BytesIO(workbook), source="spreadsheet_block"
                        )
                        check_xlsx_sheet_size(
                            io.BytesIO(workbook),
                            input_data.sheet_name,
                            source="spreadsheet_block",
                        )
                        # dtype=object and keep_default_na=False keep the sheet's own
                        # values: "N/A"/"NULL"/"NaN" stay text instead of being read as
                        # missing, and a gap in a numeric column no longer upcasts the
                        # whole column to float (turning 12 into "12.0").
                        # header=None hands every row, header included, to the CSV
                        # reader below, so pandas doesn't rename duplicate or blank
                        # headers ("a.1", "Unnamed: 2") and Excel gets the same
                        # header handling as CSV.
                        df = pd.read_excel(
                            io.BytesIO(workbook),
                            sheet_name=input_data.sheet_name or 0,
                            header=None,
                            dtype=object,
                            keep_default_na=False,
                        )
                        # Write the CSV with the same dialect it is parsed with below,
                        # otherwise a non-default delimiter yields a single column.
                        csv_buffer = _BoundedTextBuffer(MAX_FILE_SIZE_BYTES)
                        df.to_csv(
                            csv_buffer,
                            index=False,
                            header=False,
                            sep=input_data.delimiter,
                            quotechar=input_data.quotechar,
                        )
                        return csv_buffer.getvalue()

                    csv_content = await asyncio.to_thread(_read_xlsx)

                except ImportError:
                    raise ValueError(
                        "pandas library is required to read Excel files. Please install it."
                    )
                except Exception as e:
                    raise ValueError(f"Unable to read Excel file: {e}")
            else:
                # Handle CSV/text files
                csv_content = Path(file_path).read_text(encoding="utf-8")
        elif input_data.contents:
            # Use direct string content
            csv_content = input_data.contents
        else:
            raise ValueError("Either 'contents' or 'file_input' must be provided")

        csv_file = StringIO(csv_content)
        reader = csv.reader(
            csv_file,
            delimiter=input_data.delimiter,
            quotechar=input_data.quotechar,
            escapechar=input_data.escapechar,
        )

        # Blank lines before the header count toward skip_rows. A header whose
        # cells are all empty is still the header: skipping it would make the
        # first data row the header and lose its values.
        skip_rows = input_data.skip_rows
        header = None
        if input_data.has_header:
            # Empty input has no header row; don't let StopIteration escape
            # the async generator (it becomes a RuntimeError).
            header = next(reader, None)
            while header == []:
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

        rows = _collect_rows(reader, process_row)

        if input_data.produce_singular_result:
            for processed_row in rows:
                yield "row", processed_row
        else:
            yield "rows", rows


def _collect_rows(
    reader: Iterable[list[str]], process_row: Callable[[list[str]], dict[str, str]]
) -> list[dict[str, str]]:
    """Process every row, refusing once the output passes the rows limits."""
    rows = []
    size = names_size = 0
    for row in reader:
        # csv.reader yields [] for a blank line; don't turn those into {} rows.
        if not row:
            continue
        processed = process_row(row)
        names = sum(_utf8_size(name) for name in processed)
        values = sum(_utf8_size(value) for value in processed.values())
        names_size += names
        size += ROWS_ITEM_SIZE * (1 + len(processed)) + names + values
        if size > MAX_ROWS_SIZE or names_size > MAX_ROWS_NAMES_SIZE:
            raise ValueError(
                "The spreadsheet is too large for this block to output; read a "
                "smaller file, or skip some columns."
            )
        rows.append(processed)
    return rows


def _utf8_size(text: str) -> int:
    return len(text) if text.isascii() else len(text.encode())


class _BoundedTextBuffer(io.StringIO):
    """A text buffer that refuses to grow past ``limit`` characters."""

    def __init__(self, limit: int):
        super().__init__()
        self._limit = limit

    def write(self, text: str) -> int:
        if self.tell() + len(text) > self._limit:
            raise ValueError(
                f"The sheet is larger than {self._limit // (1024 * 1024)} MB as "
                "CSV, more than can be read."
            )
        return super().write(text)


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
