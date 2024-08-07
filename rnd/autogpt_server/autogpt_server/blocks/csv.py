from autogpt_server.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from autogpt_server.data.model import ContributorDetails


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
        data: dict[str, str]

    def __init__(self):
        super().__init__(
            id="acf7625e-d2cb-4941-bfeb-2819fc6fc015",
            input_schema=ReadCsvBlock.Input,
            output_schema=ReadCsvBlock.Output,
            contributors=[ContributorDetails(name="Nicholas Tindle")],
            categories={BlockCategory.TEXT},
            test_input={
                "contents": "a, b, c\n1,2,3\n4,5,6",
            },
            test_output=[
                ("data", {"a": "1", "b": "2", "c": "3"}),
                ("data", {"a": "4", "b": "5", "c": "6"}),
            ],
        )

    def run(self, input_data: Input) -> BlockOutput:
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

        # join the data with the header
        for row in reader:
            data = {}
            for i, value in enumerate(row):
                if i not in input_data.skip_columns:
                    if input_data.has_header and header:
                        data[header[i]] = value.strip() if input_data.strip else value
                    else:
                        data[str(i)] = value.strip() if input_data.strip else value
            yield "data", data
