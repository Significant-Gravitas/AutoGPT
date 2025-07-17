from pathlib import Path

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import ContributorDetails, SchemaField
from backend.util.file import get_exec_file_path, store_media_file
from backend.util.type import MediaFileType


class ReadSpreadsheetBlock(Block):
    class Input(BlockSchema):
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

    class Output(BlockSchema):
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
        )

    async def run(
        self, input_data: Input, *, graph_exec_id: str, user_id: str, **_kwargs
    ) -> BlockOutput:
        import csv
        from io import StringIO

        # Determine data source - prefer file_input if provided, otherwise use contents
        if input_data.file_input:
            stored_file_path = await store_media_file(
                user_id=user_id,
                graph_exec_id=graph_exec_id,
                file=input_data.file_input,
                return_content=False,
            )

            # Get full file path
            file_path = get_exec_file_path(graph_exec_id, stored_file_path)
            if not Path(file_path).exists():
                raise ValueError(f"File does not exist: {file_path}")

            # Check if file is an Excel file and convert to CSV
            file_extension = Path(file_path).suffix.lower()

            if file_extension in [".xlsx", ".xls"]:
                # Handle Excel files
                try:
                    from io import StringIO

                    import pandas as pd

                    # Read Excel file
                    df = pd.read_excel(file_path)

                    # Convert to CSV string
                    csv_buffer = StringIO()
                    df.to_csv(csv_buffer, index=False)
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
                if i not in input_data.skip_columns:
                    if input_data.has_header and header:
                        data[header[i]] = value.strip() if input_data.strip else value
                    else:
                        data[str(i)] = value.strip() if input_data.strip else value
            return data

        rows = [process_row(row) for row in reader]

        if input_data.produce_singular_result:
            for processed_row in rows:
                yield "row", processed_row
        else:
            yield "rows", rows
