from typing import Any, List

from backend.blocks._base import (
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
            description="A block that looks up a value in a dictionary, list, or object by key or index and returns the corresponding value.",
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


# =============================================================================
# List Concatenation Helpers
# =============================================================================


def _validate_list_input(item: Any, index: int) -> str | None:
    """Validate that an item is a list. Returns error message or None."""
    if item is None:
        return None  # None is acceptable, will be skipped
    if not isinstance(item, list):
        return (
            f"Invalid input at index {index}: expected a list, "
            f"got {type(item).__name__}. "
            f"All items in 'lists' must be lists (e.g., [[1, 2], [3, 4]])."
        )
    return None


def _validate_all_lists(lists: List[Any]) -> str | None:
    """Validate that all items in a sequence are lists. Returns first error or None."""
    for idx, item in enumerate(lists):
        error = _validate_list_input(item, idx)
        if error is not None and item is not None:
            return error
    return None


def _concatenate_lists_simple(lists: List[List[Any]]) -> List[Any]:
    """Concatenate a sequence of lists into a single list, skipping None values."""
    result: List[Any] = []
    for lst in lists:
        if lst is None:
            continue
        result.extend(lst)
    return result


def _flatten_nested_list(nested: List[Any], max_depth: int = -1) -> List[Any]:
    """
    Recursively flatten a nested list structure.

    Args:
        nested: The list to flatten.
        max_depth: Maximum recursion depth. -1 means unlimited.

    Returns:
        A flat list with all nested elements extracted.
    """
    result: List[Any] = []
    _flatten_recursive(nested, result, current_depth=0, max_depth=max_depth)
    return result


_MAX_FLATTEN_DEPTH = 1000


def _flatten_recursive(
    items: List[Any],
    result: List[Any],
    current_depth: int,
    max_depth: int,
) -> None:
    """Internal recursive helper for flattening nested lists."""
    if current_depth > _MAX_FLATTEN_DEPTH:
        raise RecursionError(
            f"Flattening exceeded maximum depth of {_MAX_FLATTEN_DEPTH} levels. "
            "Input may be too deeply nested."
        )
    for item in items:
        if isinstance(item, list) and (max_depth == -1 or current_depth < max_depth):
            _flatten_recursive(item, result, current_depth + 1, max_depth)
        else:
            result.append(item)


def _deduplicate_list(items: List[Any]) -> List[Any]:
    """
    Remove duplicate elements from a list, preserving order of first occurrences.

    Args:
        items: The list to deduplicate.

    Returns:
        A list with duplicates removed, maintaining original order.
    """
    seen: set = set()
    result: List[Any] = []
    for item in items:
        item_id = _make_hashable(item)
        if item_id not in seen:
            seen.add(item_id)
            result.append(item)
    return result


def _make_hashable(item: Any):
    """
    Create a hashable representation of any item for deduplication.
    Converts unhashable types (dicts, lists) into deterministic tuple structures.
    """
    if isinstance(item, dict):
        return tuple(
            sorted(
                ((_make_hashable(k), _make_hashable(v)) for k, v in item.items()),
                key=lambda x: (str(type(x[0])), str(x[0])),
            )
        )
    if isinstance(item, (list, tuple)):
        return tuple(_make_hashable(i) for i in item)
    if isinstance(item, set):
        return frozenset(_make_hashable(i) for i in item)
    return item


def _filter_none_values(items: List[Any]) -> List[Any]:
    """Remove None values from a list."""
    return [item for item in items if item is not None]


def _compute_nesting_depth(
    items: Any, current: int = 0, max_depth: int = _MAX_FLATTEN_DEPTH
) -> int:
    """
    Compute the maximum nesting depth of a list structure using iteration to avoid RecursionError.

    Uses a stack-based approach to handle deeply nested structures without hitting Python's
    recursion limit (~1000 levels).
    """
    if not isinstance(items, list):
        return current

    # Stack contains tuples of (item, depth)
    stack = [(items, current)]
    max_observed_depth = current

    while stack:
        item, depth = stack.pop()

        if depth > max_depth:
            return depth

        if not isinstance(item, list):
            max_observed_depth = max(max_observed_depth, depth)
            continue

        if len(item) == 0:
            max_observed_depth = max(max_observed_depth, depth + 1)
            continue

        # Add all children to stack with incremented depth
        for child in item:
            stack.append((child, depth + 1))

    return max_observed_depth


def _interleave_lists(lists: List[List[Any]]) -> List[Any]:
    """
    Interleave elements from multiple lists in round-robin fashion.
    Example: [[1,2,3], [a,b], [x,y,z]] -> [1, a, x, 2, b, y, 3, z]
    """
    if not lists:
        return []
    filtered = [lst for lst in lists if lst is not None]
    if not filtered:
        return []
    result: List[Any] = []
    max_len = max(len(lst) for lst in filtered)
    for i in range(max_len):
        for lst in filtered:
            if i < len(lst):
                result.append(lst[i])
    return result


# =============================================================================
# List Concatenation Blocks
# =============================================================================


class ConcatenateListsBlock(Block):
    """
    Concatenates two or more lists into a single list.

    This block accepts a list of lists and combines all their elements
    in order into one flat output list. It supports options for
    deduplication and None-filtering to provide flexible list merging
    capabilities for workflow pipelines.
    """

    class Input(BlockSchemaInput):
        lists: List[List[Any]] = SchemaField(
            description="A list of lists to concatenate together. All lists will be combined in order into a single list.",
            placeholder="e.g., [[1, 2], [3, 4], [5, 6]]",
        )
        deduplicate: bool = SchemaField(
            description="If True, remove duplicate elements from the concatenated result while preserving order.",
            default=False,
            advanced=True,
        )
        remove_none: bool = SchemaField(
            description="If True, remove None values from the concatenated result.",
            default=False,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        concatenated_list: List[Any] = SchemaField(
            description="The concatenated list containing all elements from all input lists in order."
        )
        length: int = SchemaField(
            description="The total number of elements in the concatenated list."
        )
        error: str = SchemaField(
            description="Error message if concatenation failed due to invalid input types."
        )

    def __init__(self):
        super().__init__(
            id="3cf9298b-5817-4141-9d80-7c2cc5199c8e",
            description="Concatenates multiple lists into a single list. All elements from all input lists are combined in order. Supports optional deduplication and None removal.",
            categories={BlockCategory.BASIC},
            input_schema=ConcatenateListsBlock.Input,
            output_schema=ConcatenateListsBlock.Output,
            test_input=[
                {"lists": [[1, 2, 3], [4, 5, 6]]},
                {"lists": [["a", "b"], ["c"], ["d", "e", "f"]]},
                {"lists": [[1, 2], []]},
                {"lists": []},
                {"lists": [[1, 2, 2, 3], [3, 4]], "deduplicate": True},
                {"lists": [[1, None, 2], [None, 3]], "remove_none": True},
            ],
            test_output=[
                ("concatenated_list", [1, 2, 3, 4, 5, 6]),
                ("length", 6),
                ("concatenated_list", ["a", "b", "c", "d", "e", "f"]),
                ("length", 6),
                ("concatenated_list", [1, 2]),
                ("length", 2),
                ("concatenated_list", []),
                ("length", 0),
                ("concatenated_list", [1, 2, 3, 4]),
                ("length", 4),
                ("concatenated_list", [1, 2, 3]),
                ("length", 3),
            ],
        )

    def _validate_inputs(self, lists: List[Any]) -> str | None:
        return _validate_all_lists(lists)

    def _perform_concatenation(self, lists: List[List[Any]]) -> List[Any]:
        return _concatenate_lists_simple(lists)

    def _apply_deduplication(self, items: List[Any]) -> List[Any]:
        return _deduplicate_list(items)

    def _apply_none_removal(self, items: List[Any]) -> List[Any]:
        return _filter_none_values(items)

    def _post_process(
        self, items: List[Any], deduplicate: bool, remove_none: bool
    ) -> List[Any]:
        """Apply all post-processing steps to the concatenated result."""
        result = items
        if remove_none:
            result = self._apply_none_removal(result)
        if deduplicate:
            result = self._apply_deduplication(result)
        return result

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        # Validate all inputs are lists
        validation_error = self._validate_inputs(input_data.lists)
        if validation_error is not None:
            yield "error", validation_error
            return

        # Perform concatenation
        concatenated = self._perform_concatenation(input_data.lists)

        # Apply post-processing
        result = self._post_process(
            concatenated, input_data.deduplicate, input_data.remove_none
        )

        yield "concatenated_list", result
        yield "length", len(result)


class FlattenListBlock(Block):
    """
    Flattens a nested list structure into a single flat list.

    This block takes a list that may contain nested lists at any depth
    and produces a single-level list with all leaf elements. Useful
    for normalizing data structures from multiple sources that may
    have varying levels of nesting.
    """

    class Input(BlockSchemaInput):
        nested_list: List[Any] = SchemaField(
            description="A potentially nested list to flatten into a single-level list.",
            placeholder="e.g., [[1, [2, 3]], [4, [5, [6]]]]",
        )
        max_depth: int = SchemaField(
            description="Maximum depth to flatten. -1 means flatten completely. 1 means flatten only one level.",
            default=-1,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        flattened_list: List[Any] = SchemaField(
            description="The flattened list with all nested elements extracted."
        )
        length: int = SchemaField(
            description="The number of elements in the flattened list."
        )
        original_depth: int = SchemaField(
            description="The maximum nesting depth of the original input list."
        )
        error: str = SchemaField(description="Error message if flattening failed.")

    def __init__(self):
        super().__init__(
            id="cc45bb0f-d035-4756-96a7-fe3e36254b4d",
            description="Flattens a nested list structure into a single flat list. Supports configurable maximum flattening depth.",
            categories={BlockCategory.BASIC},
            input_schema=FlattenListBlock.Input,
            output_schema=FlattenListBlock.Output,
            test_input=[
                {"nested_list": [[1, 2], [3, [4, 5]]]},
                {"nested_list": [1, [2, [3, [4]]]]},
                {"nested_list": [1, [2, [3, [4]]], 5], "max_depth": 1},
                {"nested_list": []},
                {"nested_list": [1, 2, 3]},
            ],
            test_output=[
                ("flattened_list", [1, 2, 3, 4, 5]),
                ("length", 5),
                ("original_depth", 3),
                ("flattened_list", [1, 2, 3, 4]),
                ("length", 4),
                ("original_depth", 4),
                ("flattened_list", [1, 2, [3, [4]], 5]),
                ("length", 4),
                ("original_depth", 4),
                ("flattened_list", []),
                ("length", 0),
                ("original_depth", 1),
                ("flattened_list", [1, 2, 3]),
                ("length", 3),
                ("original_depth", 1),
            ],
        )

    def _compute_depth(self, items: List[Any]) -> int:
        """Compute the nesting depth of the input list."""
        return _compute_nesting_depth(items)

    def _flatten(self, items: List[Any], max_depth: int) -> List[Any]:
        """Flatten the list to the specified depth."""
        return _flatten_nested_list(items, max_depth=max_depth)

    def _validate_max_depth(self, max_depth: int) -> str | None:
        """Validate the max_depth parameter."""
        if max_depth < -1:
            return f"max_depth must be -1 (unlimited) or a non-negative integer, got {max_depth}"
        return None

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        # Validate max_depth
        depth_error = self._validate_max_depth(input_data.max_depth)
        if depth_error is not None:
            yield "error", depth_error
            return

        original_depth = self._compute_depth(input_data.nested_list)
        flattened = self._flatten(input_data.nested_list, input_data.max_depth)

        yield "flattened_list", flattened
        yield "length", len(flattened)
        yield "original_depth", original_depth


class InterleaveListsBlock(Block):
    """
    Interleaves elements from multiple lists in round-robin fashion.

    Given multiple input lists, this block takes one element from each
    list in turn, producing an output where elements alternate between
    sources. Lists of different lengths are handled gracefully - shorter
    lists simply stop contributing once exhausted.
    """

    class Input(BlockSchemaInput):
        lists: List[List[Any]] = SchemaField(
            description="A list of lists to interleave. Elements will be taken in round-robin order.",
            placeholder="e.g., [[1, 2, 3], ['a', 'b', 'c']]",
        )

    class Output(BlockSchemaOutput):
        interleaved_list: List[Any] = SchemaField(
            description="The interleaved list with elements alternating from each input list."
        )
        length: int = SchemaField(
            description="The total number of elements in the interleaved list."
        )
        error: str = SchemaField(description="Error message if interleaving failed.")

    def __init__(self):
        super().__init__(
            id="9f616084-1d9f-4f8e-bc00-5b9d2a75cd75",
            description="Interleaves elements from multiple lists in round-robin fashion, alternating between sources.",
            categories={BlockCategory.BASIC},
            input_schema=InterleaveListsBlock.Input,
            output_schema=InterleaveListsBlock.Output,
            test_input=[
                {"lists": [[1, 2, 3], ["a", "b", "c"]]},
                {"lists": [[1, 2, 3], ["a", "b"], ["x", "y", "z"]]},
                {"lists": [[1], [2], [3]]},
                {"lists": []},
            ],
            test_output=[
                ("interleaved_list", [1, "a", 2, "b", 3, "c"]),
                ("length", 6),
                ("interleaved_list", [1, "a", "x", 2, "b", "y", 3, "z"]),
                ("length", 8),
                ("interleaved_list", [1, 2, 3]),
                ("length", 3),
                ("interleaved_list", []),
                ("length", 0),
            ],
        )

    def _validate_inputs(self, lists: List[Any]) -> str | None:
        return _validate_all_lists(lists)

    def _interleave(self, lists: List[List[Any]]) -> List[Any]:
        return _interleave_lists(lists)

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        validation_error = self._validate_inputs(input_data.lists)
        if validation_error is not None:
            yield "error", validation_error
            return

        result = self._interleave(input_data.lists)
        yield "interleaved_list", result
        yield "length", len(result)


class ZipListsBlock(Block):
    """
    Zips multiple lists together into a list of grouped tuples/lists.

    Takes two or more input lists and combines corresponding elements
    into sub-lists. For example, zipping [1,2,3] and ['a','b','c']
    produces [[1,'a'], [2,'b'], [3,'c']]. Supports both truncating
    to shortest list and padding to longest list with a fill value.
    """

    class Input(BlockSchemaInput):
        lists: List[List[Any]] = SchemaField(
            description="A list of lists to zip together. Corresponding elements will be grouped.",
            placeholder="e.g., [[1, 2, 3], ['a', 'b', 'c']]",
        )
        pad_to_longest: bool = SchemaField(
            description="If True, pad shorter lists with fill_value to match the longest list. If False, truncate to shortest.",
            default=False,
            advanced=True,
        )
        fill_value: Any = SchemaField(
            description="Value to use for padding when pad_to_longest is True.",
            default=None,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        zipped_list: List[List[Any]] = SchemaField(
            description="The zipped list of grouped elements."
        )
        length: int = SchemaField(
            description="The number of groups in the zipped result."
        )
        error: str = SchemaField(description="Error message if zipping failed.")

    def __init__(self):
        super().__init__(
            id="0d0e684f-5cb9-4c4b-b8d1-47a0860e0c07",
            description="Zips multiple lists together into a list of grouped elements. Supports padding to longest or truncating to shortest.",
            categories={BlockCategory.BASIC},
            input_schema=ZipListsBlock.Input,
            output_schema=ZipListsBlock.Output,
            test_input=[
                {"lists": [[1, 2, 3], ["a", "b", "c"]]},
                {"lists": [[1, 2, 3], ["a", "b"]]},
                {
                    "lists": [[1, 2], ["a", "b", "c"]],
                    "pad_to_longest": True,
                    "fill_value": 0,
                },
                {"lists": []},
            ],
            test_output=[
                ("zipped_list", [[1, "a"], [2, "b"], [3, "c"]]),
                ("length", 3),
                ("zipped_list", [[1, "a"], [2, "b"]]),
                ("length", 2),
                ("zipped_list", [[1, "a"], [2, "b"], [0, "c"]]),
                ("length", 3),
                ("zipped_list", []),
                ("length", 0),
            ],
        )

    def _validate_inputs(self, lists: List[Any]) -> str | None:
        return _validate_all_lists(lists)

    def _zip_truncate(self, lists: List[List[Any]]) -> List[List[Any]]:
        """Zip lists, truncating to shortest."""
        filtered = [lst for lst in lists if lst is not None]
        if not filtered:
            return []
        return [list(group) for group in zip(*filtered)]

    def _zip_pad(self, lists: List[List[Any]], fill_value: Any) -> List[List[Any]]:
        """Zip lists, padding shorter ones with fill_value."""
        if not lists:
            return []
        lists = [lst for lst in lists if lst is not None]
        if not lists:
            return []
        max_len = max(len(lst) for lst in lists)
        result: List[List[Any]] = []
        for i in range(max_len):
            group: List[Any] = []
            for lst in lists:
                if i < len(lst):
                    group.append(lst[i])
                else:
                    group.append(fill_value)
            result.append(group)
        return result

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        validation_error = self._validate_inputs(input_data.lists)
        if validation_error is not None:
            yield "error", validation_error
            return

        if not input_data.lists:
            yield "zipped_list", []
            yield "length", 0
            return

        if input_data.pad_to_longest:
            result = self._zip_pad(input_data.lists, input_data.fill_value)
        else:
            result = self._zip_truncate(input_data.lists)

        yield "zipped_list", result
        yield "length", len(result)


class ListDifferenceBlock(Block):
    """
    Computes the difference between two lists (elements in the first
    list that are not in the second list).

    This is useful for finding items that exist in one dataset but
    not in another, such as finding new items, missing items, or
    items that need to be processed.
    """

    class Input(BlockSchemaInput):
        list_a: List[Any] = SchemaField(
            description="The primary list to check elements from.",
            placeholder="e.g., [1, 2, 3, 4, 5]",
        )
        list_b: List[Any] = SchemaField(
            description="The list to subtract. Elements found here will be removed from list_a.",
            placeholder="e.g., [3, 4, 5, 6]",
        )
        symmetric: bool = SchemaField(
            description="If True, compute symmetric difference (elements in either list but not both).",
            default=False,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        difference: List[Any] = SchemaField(
            description="Elements from list_a not found in list_b (or symmetric difference if enabled)."
        )
        length: int = SchemaField(
            description="The number of elements in the difference result."
        )
        error: str = SchemaField(description="Error message if the operation failed.")

    def __init__(self):
        super().__init__(
            id="05309873-9d61-447e-96b5-b804e2511829",
            description="Computes the difference between two lists. Returns elements in the first list not found in the second, or symmetric difference.",
            categories={BlockCategory.BASIC},
            input_schema=ListDifferenceBlock.Input,
            output_schema=ListDifferenceBlock.Output,
            test_input=[
                {"list_a": [1, 2, 3, 4, 5], "list_b": [3, 4, 5, 6, 7]},
                {
                    "list_a": [1, 2, 3, 4, 5],
                    "list_b": [3, 4, 5, 6, 7],
                    "symmetric": True,
                },
                {"list_a": ["a", "b", "c"], "list_b": ["b"]},
                {"list_a": [], "list_b": [1, 2, 3]},
            ],
            test_output=[
                ("difference", [1, 2]),
                ("length", 2),
                ("difference", [1, 2, 6, 7]),
                ("length", 4),
                ("difference", ["a", "c"]),
                ("length", 2),
                ("difference", []),
                ("length", 0),
            ],
        )

    def _compute_difference(self, list_a: List[Any], list_b: List[Any]) -> List[Any]:
        """Compute elements in list_a not in list_b."""
        b_hashes = {_make_hashable(item) for item in list_b}
        return [item for item in list_a if _make_hashable(item) not in b_hashes]

    def _compute_symmetric_difference(
        self, list_a: List[Any], list_b: List[Any]
    ) -> List[Any]:
        """Compute elements in either list but not both."""
        a_hashes = {_make_hashable(item) for item in list_a}
        b_hashes = {_make_hashable(item) for item in list_b}
        only_in_a = [item for item in list_a if _make_hashable(item) not in b_hashes]
        only_in_b = [item for item in list_b if _make_hashable(item) not in a_hashes]
        return only_in_a + only_in_b

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        if input_data.symmetric:
            result = self._compute_symmetric_difference(
                input_data.list_a, input_data.list_b
            )
        else:
            result = self._compute_difference(input_data.list_a, input_data.list_b)

        yield "difference", result
        yield "length", len(result)


class ListIntersectionBlock(Block):
    """
    Computes the intersection of two lists (elements present in both lists).

    This is useful for finding common items between two datasets,
    such as shared tags, mutual connections, or overlapping categories.
    """

    class Input(BlockSchemaInput):
        list_a: List[Any] = SchemaField(
            description="The first list to intersect.",
            placeholder="e.g., [1, 2, 3, 4, 5]",
        )
        list_b: List[Any] = SchemaField(
            description="The second list to intersect.",
            placeholder="e.g., [3, 4, 5, 6, 7]",
        )

    class Output(BlockSchemaOutput):
        intersection: List[Any] = SchemaField(
            description="Elements present in both list_a and list_b."
        )
        length: int = SchemaField(
            description="The number of elements in the intersection."
        )
        error: str = SchemaField(description="Error message if the operation failed.")

    def __init__(self):
        super().__init__(
            id="b6eb08b6-dbe3-411b-b9b4-2508cb311a1f",
            description="Computes the intersection of two lists, returning only elements present in both.",
            categories={BlockCategory.BASIC},
            input_schema=ListIntersectionBlock.Input,
            output_schema=ListIntersectionBlock.Output,
            test_input=[
                {"list_a": [1, 2, 3, 4, 5], "list_b": [3, 4, 5, 6, 7]},
                {"list_a": ["a", "b", "c"], "list_b": ["c", "d", "e"]},
                {"list_a": [1, 2], "list_b": [3, 4]},
                {"list_a": [], "list_b": [1, 2, 3]},
            ],
            test_output=[
                ("intersection", [3, 4, 5]),
                ("length", 3),
                ("intersection", ["c"]),
                ("length", 1),
                ("intersection", []),
                ("length", 0),
                ("intersection", []),
                ("length", 0),
            ],
        )

    def _compute_intersection(self, list_a: List[Any], list_b: List[Any]) -> List[Any]:
        """Compute elements present in both lists, preserving order from list_a."""
        b_hashes = {_make_hashable(item) for item in list_b}
        seen: set = set()
        result: List[Any] = []
        for item in list_a:
            h = _make_hashable(item)
            if h in b_hashes and h not in seen:
                result.append(item)
                seen.add(h)
        return result

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        result = self._compute_intersection(input_data.list_a, input_data.list_b)
        yield "intersection", result
        yield "length", len(result)
