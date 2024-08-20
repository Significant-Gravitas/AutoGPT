from typing import Any, List, Tuple

from autogpt_server.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from autogpt_server.data.model import Field, SchemaField


class ForEachBlock(Block):
    class Input(BlockSchema):
        items: List[Any] = SchemaField(
            description="The list of items to iterate over",
            placeholder="[1, 2, 3, 4, 5]",
        )
        return_index: bool = Field(
            description="If it should just yield the item or the item and index",
            default=True,
        )

    class Output(BlockSchema):
        item: Tuple[int, Any] | Any = SchemaField(
            description="A tuple with the index and current item in the iteration"
        )

    def __init__(self):
        super().__init__(
            id="f8e7d6c5-b4a3-2c1d-0e9f-8g7h6i5j4k3l",
            input_schema=ForEachBlock.Input,
            output_schema=ForEachBlock.Output,
            categories={BlockCategory.LOGIC},
            test_input={"items": [1, "two", {"three": 3}, [4, 5]]},
            test_output=[
                ("item", (0, 1)),
                ("item", (1, "two")),
                ("item", (2, {"three": 3})),
                ("item", (3, [4, 5])),
            ],
        )

    def run(self, input_data: Input) -> BlockOutput:
        for index, item in enumerate(input_data.items):
            if input_data.return_index:
                yield "item", (index, item)
            else:
                yield "item", item
