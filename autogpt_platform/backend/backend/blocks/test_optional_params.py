from enum import Enum
from typing import List

from praw.models.reddit.mixins import Optional

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField


class SelectOptions(Enum):
    OPTION1 = "option1"
    OPTION2 = "option2"
    OPTION3 = "option3"
    OPTION4 = "option4"


class OptionalFieldsBlock(Block):
    class Input(BlockSchema):
        optional_multiselect: List[SelectOptions] | None = SchemaField(
            description="An optional multi-select input",
            enum=SelectOptions,
            default=None,
        )
        optional_select: Optional[SelectOptions] = SchemaField(
            description="An optional select input", enum=SelectOptions, default=None
        )
        optional_string: str | None = SchemaField(
            description="An optional string input", default=None
        )
        optional_list: List[str] | None = SchemaField(
            description="An optional object input", default=None
        )
        optional_int: int | None = SchemaField(
            description="An optional int input", default=12
        )

    class Output(BlockSchema):
        status: str = SchemaField(
            description="The status of the optional fields block operation."
        )
        inputs_dict: dict = SchemaField(description="nothing")

    def __init__(self):
        super().__init__(
            id="a2b3c4d5-e6f7-8g9h-i0j1-k2l3m4n5o6p7",
            description="A block demonstrating various optional input fields.",
            categories={BlockCategory.BASIC},
            input_schema=OptionalFieldsBlock.Input,
            output_schema=OptionalFieldsBlock.Output,
            test_input={
                "optional_multiselect": [SelectOptions.OPTION1, SelectOptions.OPTION2],
                "optional_string": "test string",
                "optional_select": SelectOptions.OPTION1,
                "optional_list": ["item1", "item2"],
                "optional_int": 3,
            },
            test_output=("status", "processed"),
        )

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        inputs_dict = {
            "optional_multiselect": input_data.optional_multiselect,
            "optional_select": input_data.optional_select,
            "optional_string": input_data.optional_string,
            "optional_list": input_data.optional_list,
            "optional_int": input_data.optional_int,
        }
        yield "inputs_dict", inputs_dict
        yield "status", "processed"
