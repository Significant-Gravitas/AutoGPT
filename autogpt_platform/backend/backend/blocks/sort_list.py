from typing import Any, List

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import SchemaField


class SortListBlock(Block):
    class Input(BlockSchemaInput):
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
        sorted_list: List[Any] = SchemaField(description="The sorted list.")
        length: int = SchemaField(description="The number of items in the sorted list.")
        error: str = SchemaField(description="Error message if sorting failed.")

    def __init__(self):
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
        try:
            sorted_list = _sort_items(
                input_data.list, input_data.key, input_data.reverse
            )
        except ValueError as e:
            yield "error", str(e)
            return
        except TypeError as e:
            yield "error", f"Failed to sort list: {e}"
            return

        yield "sorted_list", sorted_list
        yield "length", len(sorted_list)


def _sort_items(items: List[Any], key: str | None, reverse: bool) -> List[Any]:
    copied_items = items.copy()
    if key is None:
        return sorted(copied_items, reverse=reverse)

    for index, item in enumerate(copied_items):
        if not isinstance(item, dict):
            raise ValueError(
                f"Item at index {index} must be a dictionary when sorting by key."
            )
        if key not in item:
            raise ValueError(f"Item at index {index} is missing key '{key}'.")

    return sorted(copied_items, key=lambda item: item[key], reverse=reverse)
