import re
import json

from typing import Any
from pydantic import Field
from autogpt_server.data.block import Block, BlockOutput, BlockSchema


class TextMatcherBlock(Block):
    class Input(BlockSchema):
        text: Any = Field(description="Text to match")
        match: str = Field(description="Pattern (Regex) to match")
        data: Any = Field(description="Data to be forwarded to output")
        case_sensitive: bool = Field(description="Case sensitive match", default=True)

    class Output(BlockSchema):
        positive: Any = Field(description="Output data if match is found")
        negative: Any = Field(description="Output data if match is not found")

    def __init__(self):
        super().__init__(
            id="3060088f-6ed9-4928-9ba7-9c92823a7ccd",
            input_schema=TextMatcherBlock.Input,
            output_schema=TextMatcherBlock.Output,
        )

    def run(self, input_data: Input) -> BlockOutput:
        output = input_data.data or input_data.text
        case = 0 if input_data.case_sensitive else re.IGNORECASE
        if re.search(input_data.match, json.dumps(input_data.text), case):
            yield "positive", output
        else:
            yield "negative", output


class TextFormatterBlock(Block):
    class Input(BlockSchema):
        texts: list[str] = Field(
            description="Texts (list) to format",
            default=[]
        )
        named_texts: dict[str, str] = Field(
            description="Texts (dict) to format",
            default={}
        )
        format: str = Field(
            description="Template to format the text using `texts` and `named_texts`",
        )

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
