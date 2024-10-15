import re
from typing import Any, List

from jinja2 import BaseLoader, Environment
from pydantic import Field

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema, BlockType
from backend.data.model import SchemaField
from backend.util.mock import MockObject

jinja = Environment(loader=BaseLoader())


class StoreValueBlock(Block):
    """
    This block allows you to provide a constant value as a block, in a stateless manner.
    The common use-case is simply pass the `input` data, it will `output` the same data.
    The block output will be static, the output can be consumed multiple times.
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

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        yield "output", input_data.data or input_data.input


class PrintToConsoleBlock(Block):
    class Input(BlockSchema):
        text: str

    class Output(BlockSchema):
        status: str

    def __init__(self):
        super().__init__(
            id="f3b1c1b2-4c4f-4f0d-8d2f-4c4f0d8d2f4c",
            description="Print the given text to the console, this is used for a debugging purpose.",
            categories={BlockCategory.BASIC},
            input_schema=PrintToConsoleBlock.Input,
            output_schema=PrintToConsoleBlock.Output,
            test_input={"text": "Hello, World!"},
            test_output=("status", "printed"),
        )

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        print(">>>>> Print: ", input_data.text)
        yield "status", "printed"


class FindInDictionaryBlock(Block):
    class Input(BlockSchema):
        input: Any = Field(description="Dictionary to lookup from")
        key: str | int = Field(description="Key to lookup in the dictionary")

    class Output(BlockSchema):
        output: Any = Field(description="Value found for the given key")
        missing: Any = Field(description="Value of the input that missing the key")

    def __init__(self):
        super().__init__(
            id="0e50422c-6dee-4145-83d6-3a5a392f65de",
            description="Lookup the given key in the input dictionary/object/list and return the value.",
            input_schema=FindInDictionaryBlock.Input,
            output_schema=FindInDictionaryBlock.Output,
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
            categories={BlockCategory.BASIC},
        )

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
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


class AgentInputBlock(Block):
    """
    This block is used to provide input to the graph.

    It takes in a value, name, description, default values list and bool to limit selection to default values.

    It Outputs the value passed as input.
    """

    class Input(BlockSchema):
        value: Any = SchemaField(description="The value to be passed as input.")
        name: str = SchemaField(description="The name of the input.")
        description: str = SchemaField(
            description="The description of the input.",
            default="",
            advanced=True,
        )
        placeholder_values: List[Any] = SchemaField(
            description="The placeholder values to be passed as input.",
            default=[],
            advanced=True,
        )
        limit_to_placeholder_values: bool = SchemaField(
            description="Whether to limit the selection to placeholder values.",
            default=False,
            advanced=True,
        )

    class Output(BlockSchema):
        result: Any = SchemaField(description="The value passed as input.")

    def __init__(self):
        super().__init__(
            id="c0a8e994-ebf1-4a9c-a4d8-89d09c86741b",
            description="This block is used to provide input to the graph.",
            input_schema=AgentInputBlock.Input,
            output_schema=AgentInputBlock.Output,
            test_input=[
                {
                    "value": "Hello, World!",
                    "name": "input_1",
                    "description": "This is a test input.",
                    "placeholder_values": [],
                    "limit_to_placeholder_values": False,
                },
                {
                    "value": "Hello, World!",
                    "name": "input_2",
                    "description": "This is a test input.",
                    "placeholder_values": ["Hello, World!"],
                    "limit_to_placeholder_values": True,
                },
            ],
            test_output=[
                ("result", "Hello, World!"),
                ("result", "Hello, World!"),
            ],
            categories={BlockCategory.INPUT, BlockCategory.BASIC},
            block_type=BlockType.INPUT,
        )

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        yield "result", input_data.value


class AgentOutputBlock(Block):
    """
    Records the output of the graph for users to see.

    Attributes:
        recorded_value: The value to be recorded as output.
        name: The name of the output.
        description: The description of the output.
        fmt_string: The format string to be used to format the recorded_value.

    Outputs:
        output: The formatted recorded_value if fmt_string is provided and the recorded_value
                can be formatted, otherwise the raw recorded_value.

    Behavior:
        If fmt_string is provided and the recorded_value is of a type that can be formatted,
        the block attempts to format the recorded_value using the fmt_string.
        If formatting fails or no fmt_string is provided, the raw recorded_value is output.
    """

    class Input(BlockSchema):
        value: Any = SchemaField(description="The value to be recorded as output.")
        name: str = SchemaField(description="The name of the output.")
        description: str = SchemaField(
            description="The description of the output.",
            default="",
            advanced=True,
        )
        format: str = SchemaField(
            description="The format string to be used to format the recorded_value.",
            default="",
            advanced=True,
        )

    class Output(BlockSchema):
        output: Any = SchemaField(description="The value recorded as output.")

    def __init__(self):
        super().__init__(
            id="363ae599-353e-4804-937e-b2ee3cef3da4",
            description=("Stores the output of the graph for users to see."),
            input_schema=AgentOutputBlock.Input,
            output_schema=AgentOutputBlock.Output,
            test_input=[
                {
                    "value": "Hello, World!",
                    "name": "output_1",
                    "description": "This is a test output.",
                    "format": "{{ output_1 }}!!",
                },
                {
                    "value": "42",
                    "name": "output_2",
                    "description": "This is another test output.",
                    "format": "{{ output_2 }}",
                },
                {
                    "value": MockObject(value="!!", key="key"),
                    "name": "output_3",
                    "description": "This is a test output with a mock object.",
                    "format": "{{ output_3 }}",
                },
            ],
            test_output=[
                ("output", "Hello, World!!!"),
                ("output", "42"),
                ("output", MockObject(value="!!", key="key")),
            ],
            categories={BlockCategory.OUTPUT, BlockCategory.BASIC},
            block_type=BlockType.OUTPUT,
        )

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        """
        Attempts to format the recorded_value using the fmt_string if provided.
        If formatting fails or no fmt_string is given, returns the original recorded_value.
        """
        if input_data.format:
            try:
                fmt = re.sub(r"(?<!{){[ a-zA-Z0-9_]+}", r"{\g<0>}", input_data.format)
                template = jinja.from_string(fmt)
                yield "output", template.render({input_data.name: input_data.value})
            except Exception as e:
                yield "output", f"Error: {e}, {input_data.value}"
        else:
            yield "output", input_data.value


class AddToDictionaryBlock(Block):
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
            input_schema=AddToDictionaryBlock.Input,
            output_schema=AddToDictionaryBlock.Output,
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

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        # If no dictionary is provided, create a new one
        if input_data.dictionary is None:
            updated_dict = {}
        else:
            # Create a copy of the input dictionary to avoid modifying the original
            updated_dict = input_data.dictionary.copy()

        # Add the new key-value pair
        updated_dict[input_data.key] = input_data.value

        yield "updated_dictionary", updated_dict


class AddToListBlock(Block):
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
            input_schema=AddToListBlock.Input,
            output_schema=AddToListBlock.Output,
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

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
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

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        yield "output", input_data.text
