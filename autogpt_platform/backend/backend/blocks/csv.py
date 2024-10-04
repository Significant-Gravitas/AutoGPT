from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import ContributorDetails


class ReadCsvBlock(Block):
    class Input(BlockSchema):
        contents: str
        delimiter: str = ","
        quotechar: str = '"'
        escapechar: str = "\\"
        has_header: bool = True
        skip_rows: int = 0
        strip: bool = True
        skip_columns: list[str] = []

    class Output(BlockSchema):
        row: dict[str, str]
        all_data: list[dict[str, str]]

    def __init__(self):
        super().__init__(
            id="acf7625e-d2cb-4941-bfeb-2819fc6fc015",
            input_schema=ReadCsvBlock.Input,
            output_schema=ReadCsvBlock.Output,
            description="Reads a CSV file and outputs the data as a list of dictionaries and individual rows via rows.",
            contributors=[ContributorDetails(name="Nicholas Tindle")],
            categories={BlockCategory.TEXT, BlockCategory.DATA},
            test_input={
                "contents": "a, b, c\n1,2,3\n4,5,6",
            },
            test_output=[
                ("row", {"a": "1", "b": "2", "c": "3"}),
                ("row", {"a": "4", "b": "5", "c": "6"}),
                (
                    "all_data",
                    [
                        {"a": "1", "b": "2", "c": "3"},
                        {"a": "4", "b": "5", "c": "6"},
                    ],
                ),
            ],
        )

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        import csv
        from io import StringIO

        csv_file = StringIO(input_data.contents)
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

        all_data = []
        for row in reader:
            processed_row = process_row(row)
            all_data.append(processed_row)
            yield "row", processed_row

        yield "all_data", all_data
