from abc import ABC, abstractmethod
from typing import Any, Generic, List, TypeVar

from pydantic import Field

from autogpt_server.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from autogpt_server.data.model import SchemaField
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


class DictionaryAddEntryBlock(Block):
    class Input(BlockSchema):
        dictionary: dict | None = SchemaField(
            default=None,
            description="The dictionary to add the entry to. If not provided, a new dictionary will be created.",
            placeholder='{"key1": "value1", "key2": "value2"}',
        )
        key: str = SchemaField(
            description="The key for the new entry.", placeholder="new_key"
        )
        value: Any = SchemaField(
            description="The value for the new entry.", placeholder="new_value"
        )

    class Output(BlockSchema):
        updated_dictionary: dict = SchemaField(
            description="The dictionary with the new entry added."
        )
        error: str = SchemaField(description="Error message if the operation failed.")

    def __init__(self):
        super().__init__(
            id="31d1064e-7446-4693-a7d4-65e5ca1180d1",
            description="Adds a new key-value pair to a dictionary. If no dictionary is provided, a new one is created.",
            categories={BlockCategory.BASIC},
            input_schema=DictionaryAddEntryBlock.Input,
            output_schema=DictionaryAddEntryBlock.Output,
            test_input=[
                {
                    "dictionary": {"existing_key": "existing_value"},
                    "key": "new_key",
                    "value": "new_value",
                },
                {"key": "first_key", "value": "first_value"},
            ],
            test_output=[
                (
                    "updated_dictionary",
                    {"existing_key": "existing_value", "new_key": "new_value"},
                ),
                ("updated_dictionary", {"first_key": "first_value"}),
            ],
        )

    def run(self, input_data: Input) -> BlockOutput:
        try:
            # If no dictionary is provided, create a new one
            if input_data.dictionary is None:
                updated_dict = {}
            else:
                # Create a copy of the input dictionary to avoid modifying the original
                updated_dict = input_data.dictionary.copy()

            # Add the new key-value pair
            updated_dict[input_data.key] = input_data.value

            yield "updated_dictionary", updated_dict
        except Exception as e:
            yield "error", f"Failed to add entry to dictionary: {str(e)}"


class ListAddEntryBlock(Block):
    class Input(BlockSchema):
        list: List[Any] | None = SchemaField(
            default=None,
            description="The list to add the entry to. If not provided, a new list will be created.",
            placeholder='[1, "string", {"key": "value"}]',
        )
        entry: Any = SchemaField(
            description="The entry to add to the list. Can be of any type (string, int, dict, etc.).",
            placeholder='{"new_key": "new_value"}',
        )
        position: int | None = SchemaField(
            default=None,
            description="The position to insert the new entry. If not provided, the entry will be appended to the end of the list.",
            placeholder="0",
        )

    class Output(BlockSchema):
        updated_list: List[Any] = SchemaField(
            description="The list with the new entry added."
        )
        error: str = SchemaField(description="Error message if the operation failed.")

    def __init__(self):
        super().__init__(
            id="aeb08fc1-2fc1-4141-bc8e-f758f183a822",
            description="Adds a new entry to a list. The entry can be of any type. If no list is provided, a new one is created.",
            categories={BlockCategory.BASIC},
            input_schema=ListAddEntryBlock.Input,
            output_schema=ListAddEntryBlock.Output,
            test_input=[
                {
                    "list": [1, "string", {"existing_key": "existing_value"}],
                    "entry": {"new_key": "new_value"},
                    "position": 1,
                },
                {"entry": "first_entry"},
                {"list": ["a", "b", "c"], "entry": "d"},
            ],
            test_output=[
                (
                    "updated_list",
                    [
                        1,
                        {"new_key": "new_value"},
                        "string",
                        {"existing_key": "existing_value"},
                    ],
                ),
                ("updated_list", ["first_entry"]),
                ("updated_list", ["a", "b", "c", "d"]),
            ],
        )

    def run(self, input_data: Input) -> BlockOutput:
        try:
            # If no list is provided, create a new one
            if input_data.list is None:
                updated_list = []
            else:
                # Create a copy of the input list to avoid modifying the original
                updated_list = input_data.list.copy()

            # Add the new entry
            if input_data.position is None:
                updated_list.append(input_data.entry)
            else:
                updated_list.insert(input_data.position, input_data.entry)

            yield "updated_list", updated_list
        except Exception as e:
            yield "error", f"Failed to add entry to list: {str(e)}"
