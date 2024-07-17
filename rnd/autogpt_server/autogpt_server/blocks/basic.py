from typing import Any

from pydantic import Field

from autogpt_server.data.block import Block, BlockOutput, BlockSchema


class ValueBlock(Block):
    """
    This block allows you to provide a constant value as a block, in a stateless manner.
    The common use-case is simply pass the `input` data, it will `output` the same data.
    But this will not retain the state, once it is executed, the output is consumed.

    To retain the state, you can feed the `output` to the `data` input, so that the data
    is retained in the block for the next execution. You can then trigger the block by
    feeding the `input` pin with any data, and the block will produce value of `data`.

    Ex:
         <constant_data>  <any_trigger>
                ||           ||
       =====> `data`      `input`
      ||        \\         //
      ||          ValueBlock
      ||             ||
       ========= `output`
    """

    class Input(BlockSchema):
        input: Any = Field(
            description="Trigger the block to produce the output. "
            "The value is only used when `data` is None."
        )
        data: Any = Field(
            description="The constant data to be retained in the block. "
            "This value is passed as `output`.",
            default=None,
        )

    class Output(BlockSchema):
        output: Any

    def __init__(self):
        super().__init__(
            id="1ff065e9-88e8-4358-9d82-8dc91f622ba9",
            input_schema=ValueBlock.Input,
            output_schema=ValueBlock.Output,
            test_input=[
                {"input": "Hello, World!"},
                {"input": "Hello, World!", "data": "Existing Data"},
            ],
            test_output=[
                ("output", "Hello, World!"),  # No data provided, so trigger is returned
                ("output", "Existing Data"),  # Data is provided, so data is returned.
            ],
        )

    def run(self, input_data: Input) -> BlockOutput:
        yield "output", input_data.data or input_data.input


class PrintingBlock(Block):
    class Input(BlockSchema):
        text: str

    class Output(BlockSchema):
        status: str

    def __init__(self):
        super().__init__(
            id="f3b1c1b2-4c4f-4f0d-8d2f-4c4f0d8d2f4c",
            input_schema=PrintingBlock.Input,
            output_schema=PrintingBlock.Output,
            test_input={"text": "Hello, World!"},
            test_output=("status", "printed"),
        )

    def run(self, input_data: Input) -> BlockOutput:
        print(">>>>> Print: ", input_data.text)
        yield "status", "printed"


class ObjectLookupBlock(Block):
    class Input(BlockSchema):
        input: Any = Field(description="Dictionary to lookup from")
        key: str | int = Field(description="Key to lookup in the dictionary")

    class Output(BlockSchema):
        output: Any = Field(description="Value found for the given key")
        missing: Any = Field(description="Value of the input that missing the key")

    def __init__(self):
        super().__init__(
            id="a1b2c3d4-5e6f-7g8h-9i0j-k1l2m3n4o5p6",
            input_schema=ObjectLookupBlock.Input,
            output_schema=ObjectLookupBlock.Output,
            test_input=[
                {"input": {"apple": 1, "banana": 2, "cherry": 3}, "key": "banana"},
                {"input": {"x": 10, "y": 20, "z": 30}, "key": "w"},
                {"input": [1, 2, 3], "key": 1},
                {"input": [1, 2, 3], "key": 3},
                {"input": ObjectLookupBlock.Input(input="!!", key="key"), "key": "key"},
            ],
            test_output=[
                ("output", 2),
                ("missing", {"x": 10, "y": 20, "z": 30}),
                ("output", 2),
                ("missing", [1, 2, 3]),
                ("output", "key"),
            ],
        )

    def run(self, input_data: Input) -> BlockOutput:
        obj = input_data.input
        key = input_data.key

        if isinstance(obj, dict) and key in obj:
            yield "output", obj[key]
        elif isinstance(obj, list) and isinstance(key, int) and 0 <= key < len(obj):
            yield "output", obj[key]
        elif isinstance(obj, object) and isinstance(key, str) and hasattr(obj, key):
            yield "output", getattr(obj, key)
        else:
            yield "missing", input_data.input
