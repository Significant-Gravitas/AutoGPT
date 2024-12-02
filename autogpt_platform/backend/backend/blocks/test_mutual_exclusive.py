from typing import Optional

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField


class TestMutualExclusiveBlock(Block):
    class Input(BlockSchema):
        text_input_1: Optional[str] = SchemaField(
            title="Text Input 1",
            description="First mutually exclusive input",
            mutually_exclusive="group_1",
        )

        text_input_2: Optional[str] = SchemaField(
            title="Text Input 2",
            description="Second mutually exclusive input",
            mutually_exclusive="group_1",
        )

        text_input_3: Optional[str] = SchemaField(
            title="Text Input 3",
            description="Third mutually exclusive input",
            mutually_exclusive="group_1",
        )

        number_input_1: Optional[int] = SchemaField(
            title="Number Input 1",
            description="First number input (mutually exclusive)",
            mutually_exclusive="group2",
        )

        number_input_2: Optional[int] = SchemaField(
            title="Number Input 2",
            description="Second number input (mutually exclusive)",
            mutually_exclusive="group2",
        )

        independent_input: str = SchemaField(
            title="Independent Input",
            description="This input is not mutually exclusive with others",
            default="This can be filled anytime",
        )

    class Output(BlockSchema):
        result: str = SchemaField(description="Shows which inputs were filled")

    def __init__(self):
        super().__init__(
            id="b7faa910-b074-11ef-bee7-477f51db4711",
            description="A test block to demonstrate mutually exclusive inputs",
            categories={BlockCategory.BASIC},
            input_schema=TestMutualExclusiveBlock.Input,
            output_schema=TestMutualExclusiveBlock.Output,
        )

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        filled_inputs = []

        if input_data.text_input_1:
            filled_inputs.append(f"Text Input 1: {input_data.text_input_1}")
        if input_data.text_input_2:
            filled_inputs.append(f"Text Input 2: {input_data.text_input_2}")
        if input_data.text_input_3:
            filled_inputs.append(f"Text Input 3: {input_data.text_input_3}")

        if input_data.number_input_1:
            filled_inputs.append(f"Number Input 1: {input_data.number_input_1}")
        if input_data.number_input_2:
            filled_inputs.append(f"Number Input 2: {input_data.number_input_2}")

        filled_inputs.append(f"Independent Input: {input_data.independent_input}")

        yield "result", "\n".join(filled_inputs)
