from pathlib import Path

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.execution import ExecutionContext
from backend.data.model import ContributorDetails, SchemaField
from backend.util.file import get_exec_file_path, store_media_file
from backend.util.type import MediaFileType


class ReadSpreadsheetBlock(Block):
    class Input(BlockSchemaInput):
        contents: str | None = SchemaField(
            description="The contents of the CSV/spreadsheet data to read",
            placeholder="a, b, c\n1,2,3\n4,5,6",
            default=None,
            advanced=False,
        )
        file_input: MediaFileType | None = SchemaField(
            description=(
                "CSV or Excel file to read from (URL, data URI, or local path). "
                "Excel files are automatically converted to CSV"
            ),
            default=None,
            advanced=False,
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
            description=(
                "Column names (when has_header is True) or 0-based column indices "
                "as strings (when has_header is False) to omit from each row"
            ),
            default_factory=list,
        )
        sheet_name: str | int = SchemaField(
            description=(
                "Excel sheet to read: sheet name (e.g. 'Q2') or 0-based index. "
                "Defaults to the first sheet (0) so existing agents stay compatible. "
                "Ignored for CSV inputs."
            ),
            default=0,
            advanced=True,
        )
        produce_singular_result: bool = SchemaField(
            description=(
                "If True, yield individual 'row' outputs only (can be slow). "
                "If False, yield both 'rows' (all data)"
            ),
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
            description=(
                "Reads CSV and Excel files and outputs the data as a list of "
                "dictionaries and individual rows. Excel files are automatically "
                "converted to CSV format. Use sheet_name to select a non-first "
                "Excel sheet (default: first sheet)."
            ),
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

                    # Preserve literal NA-like strings and avoid float-upcasting
                    # integer columns that contain gaps (see #14638).
                    df = pd.read_excel(
                        file_path,
                        sheet_name=input_data.sheet_name,
                        dtype=object,
                        keep_default_na=False,
                    )

                    # Convert to CSV string using the configured delimiter so
                    # Excel and CSV branches stay consistent.
                    csv_buffer = StringIO()
                    df.to_csv(csv_buffer, index=False, sep=input_data.delimiter)
                    csv_content = csv_buffer.getvalue()

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

        header = None
        if input_data.has_header:
            header = next(reader)
            if input_data.strip:
                header = [h.strip() for h in header]

        for _ in range(input_data.skip_rows):
            next(reader)

        def process_row(row):
            data = {}
            for i, value in enumerate(row):
                if input_data.has_header and header:
                    if i >= len(header):
                        continue
                    col_key = header[i]
                    if col_key in input_data.skip_columns:
                        continue
                    data[col_key] = value.strip() if input_data.strip else value
                else:
                    if str(i) in input_data.skip_columns:
                        continue
                    data[str(i)] = value.strip() if input_data.strip else value
            return data

        rows = [process_row(row) for row in reader]

        if input_data.produce_singular_result:
            for processed_row in rows:
                yield "row", processed_row
        else:
            yield "rows", rows
