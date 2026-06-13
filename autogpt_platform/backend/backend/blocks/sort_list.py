from typing import Any, Callable, List

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import SchemaField


class SortListBlock(Block):
    """Sort a list directly or by a dictionary item key."""

    class Input(BlockSchemaInput):
        """Input schema for list sorting."""

        list: List[Any] = SchemaField(
            description="The list to sort.",
            placeholder="e.g., [3, 1, 2]",
        )
        key: str | None = SchemaField(
            default=None,
            description="Dictionary key to sort by. Leave empty to sort list items directly.",
            advanced=True,
        )
        reverse: bool = SchemaField(
            default=False,
            description="Whether to sort in descending order.",
        )

    class Output(BlockSchemaOutput):
        """Output schema for list sorting."""

        sorted_list: List[Any] = SchemaField(description="The sorted list.")
        length: int = SchemaField(description="The number of items in the sorted list.")
        error: str = SchemaField(
            default="",
            description="Error message if sorting failed.",
        )

    def __init__(self):
        """Initialize the block metadata and built-in test cases."""
        super().__init__(
            id="d294805e-3b2f-48c8-81ca-eaf13c582ef1",
            description="Sorts a list directly or by a key on dictionary items.",
            categories={BlockCategory.BASIC},
            input_schema=SortListBlock.Input,
            output_schema=SortListBlock.Output,
            test_input=[
                {"list": [3, 1, 2]},
                {"list": [3, 1, 2], "reverse": True},
                {
                    "list": [
                        {"name": "b", "score": 2},
                        {"name": "a", "score": 1},
                    ],
                    "key": "score",
                },
            ],
            test_output=[
                ("sorted_list", [1, 2, 3]),
                ("length", 3),
                ("sorted_list", [3, 2, 1]),
                ("length", 3),
                (
                    "sorted_list",
                    [
                        {"name": "a", "score": 1},
                        {"name": "b", "score": 2},
                    ],
                ),
                ("length", 2),
            ],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        """Yield sorted list outputs or a sorting error message."""
        sorted_list = _sort_items(input_data.list, input_data.key, input_data.reverse)

        yield "sorted_list", sorted_list
        yield "length", len(sorted_list)


def _sort_items(items: List[Any], key: str | None, reverse: bool) -> List[Any]:
    """Return a sorted copy of items."""
    copied_items = items.copy()
    if not key:
        return _sorted_items(copied_items, reverse=reverse)

    for index, item in enumerate(copied_items):
        if not isinstance(item, dict):
            raise ValueError(
                f"Item at index {index} must be a dictionary when sorting by key."
            )
        if key not in item:
            raise ValueError(f"Item at index {index} is missing key '{key}'.")

    return _sorted_items(copied_items, key=lambda item: item[key], reverse=reverse)


def _sorted_items(
    items: List[Any],
    reverse: bool,
    key: Callable[[Any], Any] | None = None,
) -> List[Any]:
    """Sort items and convert comparison failures into ValueError."""
    try:
        return sorted(items, key=key, reverse=reverse)
    except TypeError as e:
        raise ValueError(f"Failed to sort list: {e}") from e
