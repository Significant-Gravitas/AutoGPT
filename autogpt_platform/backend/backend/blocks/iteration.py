from typing import Any

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField
from backend.util.json import json


class StepThroughItemsBlock(Block):
    class Input(BlockSchema):
        items: list = SchemaField(
            advanced=False,
            description="The list or dictionary of items to iterate over",
            placeholder="[1, 2, 3, 4, 5] or {'key1': 'value1', 'key2': 'value2'}",
            default=[],
        )
        items_object: dict = SchemaField(
            advanced=False,
            description="The list or dictionary of items to iterate over",
            placeholder="[1, 2, 3, 4, 5] or {'key1': 'value1', 'key2': 'value2'}",
            default={},
        )
        items_str: str = SchemaField(
            advanced=False,
            description="The list or dictionary of items to iterate over",
            placeholder="[1, 2, 3, 4, 5] or {'key1': 'value1', 'key2': 'value2'}",
            default="",
        )

    class Output(BlockSchema):
        item: Any = SchemaField(description="The current item in the iteration")
        key: Any = SchemaField(
            description="The key or index of the current item in the iteration",
        )

    def __init__(self):
        super().__init__(
            id="f66a3543-28d3-4ab5-8945-9b336371e2ce",
            input_schema=StepThroughItemsBlock.Input,
            output_schema=StepThroughItemsBlock.Output,
            categories={BlockCategory.LOGIC},
            description="Iterates over a list or dictionary and outputs each item.",
            test_input={"items": [1, 2, 3, {"key1": "value1", "key2": "value2"}]},
            test_output=[
                ("item", 1),
                ("key", 0),
                ("item", 2),
                ("key", 1),
                ("item", 3),
                ("key", 2),
                ("item", {"key1": "value1", "key2": "value2"}),
                ("key", 3),
            ],
            test_mock={},
        )

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        for data in [input_data.items, input_data.items_object, input_data.items_str]:
            if not data:
                continue
            if isinstance(data, str):
                items = json.loads(data)
            else:
                items = data
            if isinstance(items, dict):
                # If items is a dictionary, iterate over its values
                for item in items.values():
                    yield "item", item
                    yield "key", item
            else:
                # If items is a list, iterate over the list
                for index, item in enumerate(items):
                    yield "item", item
                    yield "key", index
