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
            default_factory=list,
        )
        items_object: dict = SchemaField(
            advanced=False,
            description="The list or dictionary of items to iterate over",
            placeholder="[1, 2, 3, 4, 5] or {'key1': 'value1', 'key2': 'value2'}",
            default_factory=dict,
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

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        # Security fix: Add limits to prevent DoS from large iterations
        MAX_ITEMS = 10000  # Maximum items to iterate
        MAX_ITEM_SIZE = 1024 * 1024  # 1MB per item

        for data in [input_data.items, input_data.items_object, input_data.items_str]:
            if not data:
                continue

            # Limit string size before parsing
            if isinstance(data, str):
                if len(data) > MAX_ITEM_SIZE:
                    raise ValueError(
                        f"Input too large: {len(data)} bytes > {MAX_ITEM_SIZE} bytes"
                    )
                items = json.loads(data)
            else:
                items = data

            # Check total item count
            if isinstance(items, (list, dict)):
                if len(items) > MAX_ITEMS:
                    raise ValueError(f"Too many items: {len(items)} > {MAX_ITEMS}")

            iteration_count = 0
            if isinstance(items, dict):
                # If items is a dictionary, iterate over its values
                for key, value in items.items():
                    if iteration_count >= MAX_ITEMS:
                        break
                    yield "item", value
                    yield "key", key  # Fixed: should yield key, not item
                    iteration_count += 1
            else:
                # If items is a list, iterate over the list
                for index, item in enumerate(items):
                    if iteration_count >= MAX_ITEMS:
                        break
                    yield "item", item
                    yield "key", index
                    iteration_count += 1
