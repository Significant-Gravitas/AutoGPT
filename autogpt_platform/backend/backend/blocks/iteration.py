from typing import Any, Dict, List, Union

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField


class ListIteratorBlock(Block):
    class Input(BlockSchema):
        items: Union[List[Any], Dict[Any, Any]] = SchemaField(
            description="The list or dictionary of items to iterate over",
            placeholder="[1, 2, 3, 4, 5] or {'key1': 'value1', 'key2': 'value2'}",
        )

    class Output(BlockSchema):
        item: Any = SchemaField(description="The current item in the iteration")

    def __init__(self):
        super().__init__(
            id="f8e7d6c5-b4a3-2c1d-0e9f-8g7h6i5j4k3l",
            input_schema=ListIteratorBlock.Input,
            output_schema=ListIteratorBlock.Output,
            categories={BlockCategory.LOGIC},
            description="Iterates over a list or dictionary and outputs each item.",
            test_input={"items": [1, 2, 3, {"key": "value", "key2": "value2"}]},
            test_output=[
                ("item", 1),
                ("item", 2),
                ("item", 3),
                ("item", {"key": "value", "key2": "value2"}),
            ],
            test_mock={},
        )

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        items = input_data.items
        if isinstance(items, dict):
            # If items is a dictionary, iterate over its values
            for item in items.values():
                yield "item", item
        else:
            # If items is a list, iterate over the list
            for item in items:
                yield "item", item
