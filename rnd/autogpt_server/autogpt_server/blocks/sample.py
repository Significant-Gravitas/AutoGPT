# type: ignore

from typing import ClassVar

from autogpt_server.data.block import Block, BlockSchema, BlockOutput


class ParrotBlock(Block):
    id: ClassVar[str] = "1ff065e9-88e8-4358-9d82-8dc91f622ba9"

    class Input(BlockSchema):
        input: str

    class Output(BlockSchema):
        output: str

    def __init__(self):
        super().__init__(
            input_schema=ParrotBlock.Input,
            output_schema=ParrotBlock.Output,
        )

    def run(self, input_data: Input) -> BlockOutput:
        yield "output", input_data.input


class TextFormatterBlock(Block):
    id: ClassVar[str] = "db7d8f02-2f44-4c55-ab7a-eae0941f0c30"

    class Input(BlockSchema):
        texts: list[str]
        format: str

    class Output(BlockSchema):
        combined_text: str

    def __init__(self):
        super().__init__(
            input_schema=TextFormatterBlock.Input,
            output_schema=TextFormatterBlock.Output,
        )

    def run(self, input_data: Input) -> BlockOutput:
        yield "combined_text", input_data.format.format(texts=input_data.texts)


class PrintingBlock(Block):
    id: ClassVar[str] = "f3b1c1b2-4c4f-4f0d-8d2f-4c4f0d8d2f4c"

    class Input(BlockSchema):
        text: str

    class Output(BlockSchema):
        status: str

    def __init__(self):
        super().__init__(
            input_schema=PrintingBlock.Input,
            output_schema=PrintingBlock.Output,
        )

    def run(self, input_data: Input) -> BlockOutput:
        print(">>>>> Print: ", input_data.text)
        yield "status", "printed"
