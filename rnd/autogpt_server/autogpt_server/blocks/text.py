import re

from typing import Any
from autogpt_server.data.block import Block, BlockOutput, BlockSchema


class TextMatcherBlock(Block):
    class Input(BlockSchema):
        text: str
        match: str
        data: Any | None = None

    class Output(BlockSchema):
        positive: Any
        negative: Any

    def __init__(self):
        super().__init__(
            id="3060088f-6ed9-4928-9ba7-9c92823a7ccd",
            input_schema=TextMatcherBlock.Input,
            output_schema=TextMatcherBlock.Output,
        )

    def run(self, input_data: Input) -> BlockOutput:
        output = input_data.data or input_data.text
        if re.search(input_data.match, input_data.text):
            yield "positive", output
        else:
            yield "negative", output


class TextFormatterBlock(Block):
    class Input(BlockSchema):
        texts: list[str] = []
        named_texts: dict[str, str] = {}
        format: str

    class Output(BlockSchema):
        output: str

    def __init__(self):
        super().__init__(
            id="db7d8f02-2f44-4c55-ab7a-eae0941f0c30",
            input_schema=TextFormatterBlock.Input,
            output_schema=TextFormatterBlock.Output,
        )

    def run(self, input_data: Input) -> BlockOutput:
        yield "output", input_data.format.format(
            texts=input_data.texts,
            **input_data.named_texts,
        )
