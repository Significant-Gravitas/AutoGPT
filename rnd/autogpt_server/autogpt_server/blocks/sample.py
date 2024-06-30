# type: ignore

from autogpt_server.data.block import Block, BlockSchema, BlockOutput


class ParrotBlock(Block):
    class Input(BlockSchema):
        input: str

    class Output(BlockSchema):
        output: str

    def __init__(self):
        super().__init__(
            id="1ff065e9-88e8-4358-9d82-8dc91f622ba9",
            input_schema=ParrotBlock.Input,
            output_schema=ParrotBlock.Output,
        )

    def run(self, input_data: Input) -> BlockOutput:
        yield "output", input_data.input


class PrintingBlock(Block):
    class Input(BlockSchema):
        text: str

    class Output(BlockSchema):
        status: str

    def __init__(self):
        super().__init__(
            id="f3b1c1b2-4c4f-4f0d-8d2f-4c4f0d8d2f4c",
            input_schema=PrintingBlock.Input,
            output_schema=PrintingBlock.Output,
        )

    def run(self, input_data: Input) -> BlockOutput:
        print(">>>>> Print: ", input_data.text)
        yield "status", "printed"
