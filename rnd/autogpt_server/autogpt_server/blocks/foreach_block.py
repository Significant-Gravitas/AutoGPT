from typing import Any, List

from autogpt_server.data.block import Block, BlockOutput, BlockSchema
from autogpt_server.data.model import SchemaField


class ForEachBlock(Block):
    class Input(BlockSchema):
        items: List[Any] = SchemaField(
            description="The list of items to iterate over",
            placeholder="[1, 2, 3, 4, 5]",
        )

    class Output(BlockSchema):
        item: Any = SchemaField(description="The current item in the iteration")
        index: int = SchemaField(
            description="The index of the current item in the list", ge=0
        )

    def __init__(self):
        super().__init__(
            id="f8e7d6c5-b4a3-2c1d-0e9f-8g7h6i5j4k3l",
            input_schema=ForEachBlock.Input,
            output_schema=ForEachBlock.Output,
            test_input={"items": [1, "two", {"three": 3}, [4, 5]]},
            test_output=[
                ("item", 1),
                ("index", 0),
                ("item", "two"),
                ("index", 1),
                ("item", {"three": 3}),
                ("index", 2),
                ("item", [4, 5]),
                ("index", 3),
            ],
        )

    def run(self, input_data: Input) -> BlockOutput:
        for index, item in enumerate(input_data.items):
            yield "item", item
            yield "index", index
