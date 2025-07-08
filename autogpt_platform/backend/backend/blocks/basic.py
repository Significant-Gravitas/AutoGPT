import enum
from typing import Any

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema, BlockType
from backend.data.model import SchemaField
from backend.util.file import store_media_file
from backend.util.type import MediaFileType, convert


class FileStoreBlock(Block):
    class Input(BlockSchema):
        file_in: MediaFileType = SchemaField(
            description="The file to store in the temporary directory, it can be a URL, data URI, or local path."
        )
        base_64: bool = SchemaField(
            description="Whether produce an output in base64 format (not recommended, you can pass the string path just fine accross blocks).",
            default=False,
            advanced=True,
            title="Produce Base64 Output",
        )

    class Output(BlockSchema):
        file_out: MediaFileType = SchemaField(
            description="The relative path to the stored file in the temporary directory."
        )

    def __init__(self):
        super().__init__(
            id="cbb50872-625b-42f0-8203-a2ae78242d8a",
            description="Stores the input file in the temporary directory.",
            categories={BlockCategory.BASIC, BlockCategory.MULTIMEDIA},
            input_schema=FileStoreBlock.Input,
            output_schema=FileStoreBlock.Output,
            static_output=True,
        )

    async def run(
        self,
        input_data: Input,
        *,
        graph_exec_id: str,
        **kwargs,
    ) -> BlockOutput:
        yield "file_out", await store_media_file(
            graph_exec_id=graph_exec_id,
            file=input_data.file_in,
            return_content=input_data.base_64,
        )


class StoreValueBlock(Block):
    """
    This block allows you to provide a constant value as a block, in a stateless manner.
    The common use-case is simply pass the `input` data, it will `output` the same data.
    The block output will be static, the output can be consumed multiple times.
    """

    class Input(BlockSchema):
        input: Any = SchemaField(
            description="Trigger the block to produce the output. "
            "The value is only used when `data` is None."
        )
        data: Any = SchemaField(
            description="The constant data to be retained in the block. "
            "This value is passed as `output`.",
            default=None,
        )

    class Output(BlockSchema):
        output: Any = SchemaField(description="The stored data retained in the block.")

    def __init__(self):
        super().__init__(
            id="1ff065e9-88e8-4358-9d82-8dc91f622ba9",
            description="This block forwards an input value as output, allowing reuse without change.",
            categories={BlockCategory.BASIC},
            input_schema=StoreValueBlock.Input,
            output_schema=StoreValueBlock.Output,
            test_input=[
                {"input": "Hello, World!"},
                {"input": "Hello, World!", "data": "Existing Data"},
            ],
            test_output=[
                ("output", "Hello, World!"),  # No data provided, so trigger is returned
                ("output", "Existing Data"),  # Data is provided, so data is returned.
            ],
            static_output=True,
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        yield "output", input_data.data or input_data.input


class PrintToConsoleBlock(Block):
    class Input(BlockSchema):
        text: Any = SchemaField(description="The data to print to the console.")

    class Output(BlockSchema):
        output: Any = SchemaField(description="The data printed to the console.")
        status: str = SchemaField(description="The status of the print operation.")

    def __init__(self):
        super().__init__(
            id="f3b1c1b2-4c4f-4f0d-8d2f-4c4f0d8d2f4c",
            description="Print the given text to the console, this is used for a debugging purpose.",
            categories={BlockCategory.BASIC},
            input_schema=PrintToConsoleBlock.Input,
            output_schema=PrintToConsoleBlock.Output,
            test_input={"text": "Hello, World!"},
            test_output=[
                ("output", "Hello, World!"),
                ("status", "printed"),
            ],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        yield "output", input_data.text
        yield "status", "printed"


class NoteBlock(Block):
    class Input(BlockSchema):
        text: str = SchemaField(description="The text to display in the sticky note.")

    class Output(BlockSchema):
        output: str = SchemaField(description="The text to display in the sticky note.")

    def __init__(self):
        super().__init__(
            id="cc10ff7b-7753-4ff2-9af6-9399b1a7eddc",
            description="This block is used to display a sticky note with the given text.",
            categories={BlockCategory.BASIC},
            input_schema=NoteBlock.Input,
            output_schema=NoteBlock.Output,
            test_input={"text": "Hello, World!"},
            test_output=[
                ("output", "Hello, World!"),
            ],
            block_type=BlockType.NOTE,
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        yield "output", input_data.text


class TypeOptions(enum.Enum):
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    LIST = "list"
    DICTIONARY = "dictionary"


class UniversalTypeConverterBlock(Block):
    class Input(BlockSchema):
        value: Any = SchemaField(
            description="The value to convert to a universal type."
        )
        type: TypeOptions = SchemaField(description="The type to convert the value to.")

    class Output(BlockSchema):
        value: Any = SchemaField(description="The converted value.")
        error: str = SchemaField(description="Error message if conversion failed.")

    def __init__(self):
        super().__init__(
            id="95d1b990-ce13-4d88-9737-ba5c2070c97b",
            description="This block is used to convert a value to a universal type.",
            categories={BlockCategory.BASIC},
            input_schema=UniversalTypeConverterBlock.Input,
            output_schema=UniversalTypeConverterBlock.Output,
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        try:
            converted_value = convert(
                input_data.value,
                {
                    TypeOptions.STRING: str,
                    TypeOptions.NUMBER: float,
                    TypeOptions.BOOLEAN: bool,
                    TypeOptions.LIST: list,
                    TypeOptions.DICTIONARY: dict,
                }[input_data.type],
            )
            yield "value", converted_value
        except Exception as e:
            yield "error", f"Failed to convert value: {str(e)}"
