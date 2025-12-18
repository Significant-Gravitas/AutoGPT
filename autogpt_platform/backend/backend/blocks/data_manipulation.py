from typing import Any, List

from backend.data.block import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import SchemaField
from backend.util.json import loads
from backend.util.mock import MockObject
from backend.util.prompt import estimate_token_count_str

# =============================================================================
# Dictionary Manipulation Blocks
# =============================================================================


class CreateDictionaryBlock(Block):
    class Input(BlockSchemaInput):
        values: dict[str, Any] = SchemaField(
            description="Key-value pairs to create the dictionary with",
            placeholder="e.g., {'name': 'Alice', 'age': 25}",
        )

    class Output(BlockSchemaOutput):
        dictionary: dict[str, Any] = SchemaField(
            description="The created dictionary containing the specified key-value pairs"
        )
        error: str = SchemaField(
            description="Error message if dictionary creation failed"
        )

    def __init__(self):
        super().__init__(
            id="b924ddf4-de4f-4b56-9a85-358930dcbc91",
            description="Creates a dictionary with the specified key-value pairs. Use this when you know all the values you want to add upfront.",
            categories={BlockCategory.DATA},
            input_schema=CreateDictionaryBlock.Input,
            output_schema=CreateDictionaryBlock.Output,
            test_input=[
                {
                    "values": {"name": "Alice", "age": 25, "city": "New York"},
                },
                {
                    "values": {"numbers": [1, 2, 3], "active": True, "score": 95.5},
                },
            ],
            test_output=[
                (
                    "dictionary",
                    {"name": "Alice", "age": 25, "city": "New York"},
                ),
                (
                    "dictionary",
                    {"numbers": [1, 2, 3], "active": True, "score": 95.5},
                ),
            ],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        try:
            # The values are already validated by Pydantic schema
            yield "dictionary", input_data.values
        except Exception as e:
            yield "error", f"Failed to create dictionary: {str(e)}"


class AddToDictionaryBlock(Block):
    class Input(BlockSchemaInput):
        dictionary: dict[Any, Any] = SchemaField(
            default_factory=dict,
            description="The dictionary to add the entry to. If not provided, a new dictionary will be created.",
            advanced=False,
        )
        key: str = SchemaField(
            default="",
            description="The key for the new entry.",
            placeholder="new_key",
            advanced=False,
        )
        value: Any = SchemaField(
            default=None,
            description="The value for the new entry.",
            placeholder="new_value",
            advanced=False,
        )
        entries: dict[Any, Any] = SchemaField(
            default_factory=dict,
            description="The entries to add to the dictionary. This is the batch version of the `key` and `value` fields.",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        updated_dictionary: dict = SchemaField(
            description="The dictionary with the new entry added."
        )

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
                {
                    "dictionary": {"existing_key": "existing_value"},
                    "entries": {"new_key": "new_value", "first_key": "first_value"},
                },
            ],
            test_output=[
                (
                    "updated_dictionary",
                    {"existing_key": "existing_value", "new_key": "new_value"},
                ),
                ("updated_dictionary", {"first_key": "first_value"}),
                (
                    "updated_dictionary",
                    {
                        "existing_key": "existing_value",
                        "new_key": "new_value",
                        "first_key": "first_value",
                    },
                ),
            ],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        updated_dict = input_data.dictionary.copy()

        if input_data.value is not None and input_data.key:
            updated_dict[input_data.key] = input_data.value

        for key, value in input_data.entries.items():
            updated_dict[key] = value

        yield "updated_dictionary", updated_dict


class FindInDictionaryBlock(Block):
    class Input(BlockSchemaInput):
        input: Any = SchemaField(description="Dictionary to lookup from")
        key: str | int = SchemaField(description="Key to lookup in the dictionary")

    class Output(BlockSchemaOutput):
        output: Any = SchemaField(description="Value found for the given key")
        missing: Any = SchemaField(
            description="Value of the input that missing the key"
        )

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

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        obj = input_data.input
        key = input_data.key

        if isinstance(obj, str):
            obj = loads(obj)

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


class RemoveFromDictionaryBlock(Block):
    class Input(BlockSchemaInput):
        dictionary: dict[Any, Any] = SchemaField(
            description="The dictionary to modify."
        )
        key: str | int = SchemaField(description="Key to remove from the dictionary.")
        return_value: bool = SchemaField(
            default=False, description="Whether to return the removed value."
        )

    class Output(BlockSchemaOutput):
        updated_dictionary: dict[Any, Any] = SchemaField(
            description="The dictionary after removal."
        )
        removed_value: Any = SchemaField(description="The removed value if requested.")

    def __init__(self):
        super().__init__(
            id="46afe2ea-c613-43f8-95ff-6692c3ef6876",
            description="Removes a key-value pair from a dictionary.",
            categories={BlockCategory.BASIC},
            input_schema=RemoveFromDictionaryBlock.Input,
            output_schema=RemoveFromDictionaryBlock.Output,
            test_input=[
                {
                    "dictionary": {"a": 1, "b": 2, "c": 3},
                    "key": "b",
                    "return_value": True,
                },
                {"dictionary": {"x": "hello", "y": "world"}, "key": "x"},
            ],
            test_output=[
                ("updated_dictionary", {"a": 1, "c": 3}),
                ("removed_value", 2),
                ("updated_dictionary", {"y": "world"}),
            ],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        updated_dict = input_data.dictionary.copy()
        try:
            removed_value = updated_dict.pop(input_data.key)
            yield "updated_dictionary", updated_dict
            if input_data.return_value:
                yield "removed_value", removed_value
        except KeyError:
            yield "error", f"Key '{input_data.key}' not found in dictionary"


class ReplaceDictionaryValueBlock(Block):
    class Input(BlockSchemaInput):
        dictionary: dict[Any, Any] = SchemaField(
            description="The dictionary to modify."
        )
        key: str | int = SchemaField(description="Key to replace the value for.")
        value: Any = SchemaField(description="The new value for the given key.")

    class Output(BlockSchemaOutput):
        updated_dictionary: dict[Any, Any] = SchemaField(
            description="The dictionary after replacement."
        )
        old_value: Any = SchemaField(description="The value that was replaced.")

    def __init__(self):
        super().__init__(
            id="27e31876-18b6-44f3-ab97-f6226d8b3889",
            description="Replaces the value for a specified key in a dictionary.",
            categories={BlockCategory.BASIC},
            input_schema=ReplaceDictionaryValueBlock.Input,
            output_schema=ReplaceDictionaryValueBlock.Output,
            test_input=[
                {"dictionary": {"a": 1, "b": 2, "c": 3}, "key": "b", "value": 99},
                {
                    "dictionary": {"x": "hello", "y": "world"},
                    "key": "y",
                    "value": "universe",
                },
            ],
            test_output=[
                ("updated_dictionary", {"a": 1, "b": 99, "c": 3}),
                ("old_value", 2),
                ("updated_dictionary", {"x": "hello", "y": "universe"}),
                ("old_value", "world"),
            ],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        updated_dict = input_data.dictionary.copy()
        try:
            old_value = updated_dict[input_data.key]
            updated_dict[input_data.key] = input_data.value
            yield "updated_dictionary", updated_dict
            yield "old_value", old_value
        except KeyError:
            yield "error", f"Key '{input_data.key}' not found in dictionary"


class DictionaryIsEmptyBlock(Block):
    class Input(BlockSchemaInput):
        dictionary: dict[Any, Any] = SchemaField(description="The dictionary to check.")

    class Output(BlockSchemaOutput):
        is_empty: bool = SchemaField(description="True if the dictionary is empty.")

    def __init__(self):
        super().__init__(
            id="a3cf3f64-6bb9-4cc6-9900-608a0b3359b0",
            description="Checks if a dictionary is empty.",
            categories={BlockCategory.BASIC},
            input_schema=DictionaryIsEmptyBlock.Input,
            output_schema=DictionaryIsEmptyBlock.Output,
            test_input=[{"dictionary": {}}, {"dictionary": {"a": 1}}],
            test_output=[("is_empty", True), ("is_empty", False)],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        yield "is_empty", len(input_data.dictionary) == 0


# =============================================================================
# List Manipulation Blocks
# =============================================================================


class CreateListBlock(Block):
    class Input(BlockSchemaInput):
        values: List[Any] = SchemaField(
            description="A list of values to be combined into a new list.",
            placeholder="e.g., ['Alice', 25, True]",
        )
        max_size: int | None = SchemaField(
            default=None,
            description="Maximum size of the list. If provided, the list will be yielded in chunks of this size.",
            advanced=True,
        )
        max_tokens: int | None = SchemaField(
            default=None,
            description="Maximum tokens for the list. If provided, the list will be yielded in chunks that fit within this token limit.",
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        list: List[Any] = SchemaField(
            description="The created list containing the specified values."
        )

    def __init__(self):
        super().__init__(
            id="a912d5c7-6e00-4542-b2a9-8034136930e4",
            description="Creates a list with the specified values. Use this when you know all the values you want to add upfront. This block can also yield the list in batches based on a maximum size or token limit.",
            categories={BlockCategory.DATA},
            input_schema=CreateListBlock.Input,
            output_schema=CreateListBlock.Output,
            test_input=[
                {
                    "values": ["Alice", 25, True],
                },
                {
                    "values": [1, 2, 3, "four", {"key": "value"}],
                },
            ],
            test_output=[
                (
                    "list",
                    ["Alice", 25, True],
                ),
                (
                    "list",
                    [1, 2, 3, "four", {"key": "value"}],
                ),
            ],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        chunk = []
        cur_tokens, max_tokens = 0, input_data.max_tokens
        cur_size, max_size = 0, input_data.max_size

        for value in input_data.values:
            if max_tokens:
                tokens = estimate_token_count_str(value)
            else:
                tokens = 0

            # Check if adding this value would exceed either limit
            if (max_tokens and (cur_tokens + tokens > max_tokens)) or (
                max_size and (cur_size + 1 > max_size)
            ):
                yield "list", chunk
                chunk = [value]
                cur_size, cur_tokens = 1, tokens
            else:
                chunk.append(value)
                cur_size, cur_tokens = cur_size + 1, cur_tokens + tokens

        # Yield final chunk if any
        if chunk or not input_data.values:
            yield "list", chunk


class AddToListBlock(Block):
    class Input(BlockSchemaInput):
        list: List[Any] = SchemaField(
            default_factory=list,
            advanced=False,
            description="The list to add the entry to. If not provided, a new list will be created.",
        )
        entry: Any = SchemaField(
            description="The entry to add to the list. Can be of any type (string, int, dict, etc.).",
            advanced=False,
            default=None,
        )
        entries: List[Any] = SchemaField(
            default_factory=lambda: list(),
            description="The entries to add to the list. This is the batch version of the `entry` field.",
            advanced=True,
        )
        position: int | None = SchemaField(
            default=None,
            description="The position to insert the new entry. If not provided, the entry will be appended to the end of the list.",
        )

    class Output(BlockSchemaOutput):
        updated_list: List[Any] = SchemaField(
            description="The list with the new entry added."
        )

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
                {
                    "entry": "e",
                    "entries": ["f", "g"],
                    "list": ["a", "b"],
                    "position": 1,
                },
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
                ("updated_list", ["a", "f", "g", "e", "b"]),
            ],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        entries_added = input_data.entries.copy()
        if input_data.entry:
            entries_added.append(input_data.entry)

        updated_list = input_data.list.copy()
        if (pos := input_data.position) is not None:
            updated_list = updated_list[:pos] + entries_added + updated_list[pos:]
        else:
            updated_list += entries_added

        yield "updated_list", updated_list


class FindInListBlock(Block):
    class Input(BlockSchemaInput):
        list: List[Any] = SchemaField(description="The list to search in.")
        value: Any = SchemaField(description="The value to search for.")

    class Output(BlockSchemaOutput):
        index: int = SchemaField(description="The index of the value in the list.")
        found: bool = SchemaField(
            description="Whether the value was found in the list."
        )
        not_found_value: Any = SchemaField(
            description="The value that was not found in the list."
        )

    def __init__(self):
        super().__init__(
            id="5e2c6d0a-1e37-489f-b1d0-8e1812b23333",
            description="Finds the index of the value in the list.",
            categories={BlockCategory.BASIC},
            input_schema=FindInListBlock.Input,
            output_schema=FindInListBlock.Output,
            test_input=[
                {"list": [1, 2, 3, 4, 5], "value": 3},
                {"list": [1, 2, 3, 4, 5], "value": 6},
            ],
            test_output=[
                ("index", 2),
                ("found", True),
                ("found", False),
                ("not_found_value", 6),
            ],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        try:
            yield "index", input_data.list.index(input_data.value)
            yield "found", True
        except ValueError:
            yield "found", False
            yield "not_found_value", input_data.value


class GetListItemBlock(Block):
    class Input(BlockSchemaInput):
        list: List[Any] = SchemaField(description="The list to get the item from.")
        index: int = SchemaField(
            description="The 0-based index of the item (supports negative indices)."
        )

    class Output(BlockSchemaOutput):
        item: Any = SchemaField(description="The item at the specified index.")

    def __init__(self):
        super().__init__(
            id="262ca24c-1025-43cf-a578-534e23234e97",
            description="Returns the element at the given index.",
            categories={BlockCategory.BASIC},
            input_schema=GetListItemBlock.Input,
            output_schema=GetListItemBlock.Output,
            test_input=[
                {"list": [1, 2, 3], "index": 1},
                {"list": [1, 2, 3], "index": -1},
            ],
            test_output=[
                ("item", 2),
                ("item", 3),
            ],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        try:
            yield "item", input_data.list[input_data.index]
        except IndexError:
            yield "error", "Index out of range"


class RemoveFromListBlock(Block):
    class Input(BlockSchemaInput):
        list: List[Any] = SchemaField(description="The list to modify.")
        value: Any = SchemaField(
            default=None, description="Value to remove from the list."
        )
        index: int | None = SchemaField(
            default=None,
            description="Index of the item to pop (supports negative indices).",
        )
        return_item: bool = SchemaField(
            default=False, description="Whether to return the removed item."
        )

    class Output(BlockSchemaOutput):
        updated_list: List[Any] = SchemaField(description="The list after removal.")
        removed_item: Any = SchemaField(description="The removed item if requested.")

    def __init__(self):
        super().__init__(
            id="d93c5a93-ac7e-41c1-ae5c-ef67e6e9b826",
            description="Removes an item from a list by value or index.",
            categories={BlockCategory.BASIC},
            input_schema=RemoveFromListBlock.Input,
            output_schema=RemoveFromListBlock.Output,
            test_input=[
                {"list": [1, 2, 3], "index": 1, "return_item": True},
                {"list": ["a", "b", "c"], "value": "b"},
            ],
            test_output=[
                ("updated_list", [1, 3]),
                ("removed_item", 2),
                ("updated_list", ["a", "c"]),
            ],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        lst = input_data.list.copy()
        removed = None
        try:
            if input_data.index is not None:
                removed = lst.pop(input_data.index)
            elif input_data.value is not None:
                lst.remove(input_data.value)
                removed = input_data.value
            else:
                raise ValueError("No index or value provided for removal")
        except (IndexError, ValueError):
            yield "error", "Index or value not found"
            return

        yield "updated_list", lst
        if input_data.return_item:
            yield "removed_item", removed


class ReplaceListItemBlock(Block):
    class Input(BlockSchemaInput):
        list: List[Any] = SchemaField(description="The list to modify.")
        index: int = SchemaField(
            description="Index of the item to replace (supports negative indices)."
        )
        value: Any = SchemaField(description="The new value for the given index.")

    class Output(BlockSchemaOutput):
        updated_list: List[Any] = SchemaField(description="The list after replacement.")
        old_item: Any = SchemaField(description="The item that was replaced.")

    def __init__(self):
        super().__init__(
            id="fbf62922-bea1-4a3d-8bac-23587f810b38",
            description="Replaces an item at the specified index.",
            categories={BlockCategory.BASIC},
            input_schema=ReplaceListItemBlock.Input,
            output_schema=ReplaceListItemBlock.Output,
            test_input=[
                {"list": [1, 2, 3], "index": 1, "value": 99},
                {"list": ["a", "b"], "index": -1, "value": "c"},
            ],
            test_output=[
                ("updated_list", [1, 99, 3]),
                ("old_item", 2),
                ("updated_list", ["a", "c"]),
                ("old_item", "b"),
            ],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        lst = input_data.list.copy()
        try:
            old = lst[input_data.index]
            lst[input_data.index] = input_data.value
        except IndexError:
            yield "error", "Index out of range"
            return

        yield "updated_list", lst
        yield "old_item", old


class ListIsEmptyBlock(Block):
    class Input(BlockSchemaInput):
        list: List[Any] = SchemaField(description="The list to check.")

    class Output(BlockSchemaOutput):
        is_empty: bool = SchemaField(description="True if the list is empty.")

    def __init__(self):
        super().__init__(
            id="896ed73b-27d0-41be-813c-c1c1dc856c03",
            description="Checks if a list is empty.",
            categories={BlockCategory.BASIC},
            input_schema=ListIsEmptyBlock.Input,
            output_schema=ListIsEmptyBlock.Output,
            test_input=[{"list": []}, {"list": [1]}],
            test_output=[("is_empty", True), ("is_empty", False)],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        yield "is_empty", len(input_data.list) == 0
