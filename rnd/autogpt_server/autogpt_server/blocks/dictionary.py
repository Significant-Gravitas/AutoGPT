from typing import Any
from pydantic import Field
from autogpt_server.data.block import Block, BlockOutput, BlockSchema

class DictionaryLookup(Block):
    class Input(BlockSchema):
        dictionary: dict = Field(description="Dictionary to lookup from")
        key: Any = Field(description="Key to lookup in the dictionary")

    class Output(BlockSchema):
        value: Any = Field(description="Value found for the given key")
        found: bool = Field(description="Whether the key was found in the dictionary")

    def __init__(self):
        super().__init__(
            id="a1b2c3d4-5e6f-7g8h-9i0j-k1l2m3n4o5p6",
            input_schema=DictionaryLookup.Input,
            output_schema=DictionaryLookup.Output,
            test_input=[
                {"dictionary": {"apple": 1, "banana": 2, "cherry": 3}, "key": "banana"},
                {"dictionary": {"x": 10, "y": 20, "z": 30}, "key": "w"},
            ],
            test_output=[
                ("value", 2),
                ("found", True),
                ("value", None),
                ("found", False),
            ],
        )

    def run(self, input_data: Input) -> BlockOutput:
        if input_data.key in input_data.dictionary:
            yield "value", input_data.dictionary[input_data.key]
            yield "found", True
        else:
            yield "value", None
            yield "found", False