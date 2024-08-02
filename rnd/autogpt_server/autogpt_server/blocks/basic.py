from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import Field

from autogpt_server.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from autogpt_server.util.mock import MockObject


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
            description="This block forwards the `input` pin to `output` pin. "
            "If the `data` is provided, it will prioritize forwarding `data` "
            "over `input`. By connecting the `output` pin to `data` pin, "
            "you can retain a constant value for the next executions.",
            categories={BlockCategory.BASIC},
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
            description="Print the given text to the console, this is used for a debugging purpose.",
            categories={BlockCategory.BASIC},
            input_schema=PrintingBlock.Input,
            output_schema=PrintingBlock.Output,
            test_input={"text": "Hello, World!"},
            test_output=("status", "printed"),
        )

    def run(self, input_data: Input) -> BlockOutput:
        print(">>>>> Print: ", input_data.text)
        yield "status", "printed"


T = TypeVar("T")


class ObjectLookupBaseInput(BlockSchema, Generic[T]):
    input: T = Field(description="Dictionary to lookup from")
    key: str | int = Field(description="Key to lookup in the dictionary")


class ObjectLookupBaseOutput(BlockSchema, Generic[T]):
    output: T = Field(description="Value found for the given key")
    missing: T = Field(description="Value of the input that missing the key")


class ObjectLookupBase(Block, ABC, Generic[T]):
    @abstractmethod
    def block_id(self) -> str:
        pass

    def __init__(self, *args, **kwargs):
        input_schema = ObjectLookupBaseInput[T]
        output_schema = ObjectLookupBaseOutput[T]

        super().__init__(
            id=self.block_id(),
            description="Lookup the given key in the input dictionary/object/list and return the value.",
            input_schema=input_schema,
            output_schema=output_schema,
            test_input=[
                {"input": {"apple": 1, "banana": 2, "cherry": 3}, "key": "banana"},
                {"input": {"x": 10, "y": 20, "z": 30}, "key": "w"},
                {"input": [1, 2, 3], "key": 1},
                {"input": [1, 2, 3], "key": 3},
                {"input": MockObject(value="!!", key="key"), "key": "key"},
                {"input": [{"k1": "v1"}, {"k2": "v2"}, {"k1": "v3"}], "key": "k1"},
            ],
            test_output=[
                ("output", 2),
                ("missing", {"x": 10, "y": 20, "z": 30}),
                ("output", 2),
                ("missing", [1, 2, 3]),
                ("output", "key"),
                ("output", ["v1", "v3"]),
            ],
            *args,
            **kwargs,
        )

    def run(self, input_data: ObjectLookupBaseInput[T]) -> BlockOutput:
        obj = input_data.input
        key = input_data.key

        if isinstance(obj, dict) and key in obj:
            yield "output", obj[key]
        elif isinstance(obj, list) and isinstance(key, int) and 0 <= key < len(obj):
            yield "output", obj[key]
        elif isinstance(obj, list) and isinstance(key, str):
            if len(obj) == 0:
                yield "output", []
            elif isinstance(obj[0], dict) and key in obj[0]:
                yield "output", [item[key] for item in obj if key in item]
            else:
                yield "output", [getattr(val, key) for val in obj if hasattr(val, key)]
        elif isinstance(obj, object) and isinstance(key, str) and hasattr(obj, key):
            yield "output", getattr(obj, key)
        else:
            yield "missing", input_data.input


class ObjectLookupBlock(ObjectLookupBase[Any]):

    def __init__(self):
        super().__init__(categories={BlockCategory.BASIC})

    def block_id(self) -> str:
        return "b2g2c3d4-5e6f-7g8h-9i0j-k1l2m3n4o5p6"


class InputBlock(ObjectLookupBase[Any]):

    def __init__(self):
        super().__init__(categories={BlockCategory.BASIC, BlockCategory.INPUT_OUTPUT})

    def block_id(self) -> str:
        return "c0a8e994-ebf1-4a9c-a4d8-89d09c86741b"


class OutputBlock(ObjectLookupBase[Any]):

    def __init__(self):
        super().__init__(categories={BlockCategory.BASIC, BlockCategory.INPUT_OUTPUT})

    def block_id(self) -> str:
        return "363ae599-353e-4804-937e-b2ee3cef3da4"
