import json
import re
from typing import Any

from pydantic import Field

from autogpt_server.data.block import Block, BlockCategory, BlockOutput, BlockSchema


class TextMatcherBlock(Block):
    class Input(BlockSchema):
        text: Any = Field(description="Text to match")
        match: str = Field(description="Pattern (Regex) to match")
        data: Any = Field(description="Data to be forwarded to output")
        case_sensitive: bool = Field(description="Case sensitive match", default=True)
        dot_all: bool = Field(description="Dot matches all", default=True)

    class Output(BlockSchema):
        positive: Any = Field(description="Output data if match is found")
        negative: Any = Field(description="Output data if match is not found")

    def __init__(self):
        super().__init__(
            id="3060088f-6ed9-4928-9ba7-9c92823a7ccd",
            description="This block matches the given text with the pattern (regex) and"
            " forwards the provided data to positive (if matching) or"
            " negative (if not matching) output.",
            categories={BlockCategory.TEXT},
            input_schema=TextMatcherBlock.Input,
            output_schema=TextMatcherBlock.Output,
            test_input=[
                {"text": "ABC", "match": "ab", "data": "X", "case_sensitive": False},
                {"text": "ABC", "match": "ab", "data": "Y", "case_sensitive": True},
                {"text": "Hello World!", "match": ".orld.+", "data": "Z"},
                {"text": "Hello World!", "match": "World![a-z]+", "data": "Z"},
            ],
            test_output=[
                ("positive", "X"),
                ("negative", "Y"),
                ("positive", "Z"),
                ("negative", "Z"),
            ],
        )

    def run(self, input_data: Input) -> BlockOutput:
        output = input_data.data or input_data.text
        flags = 0
        if not input_data.case_sensitive:
            flags = flags | re.IGNORECASE
        if input_data.dot_all:
            flags = flags | re.DOTALL

        if isinstance(input_data.text, str):
            text = input_data.text
        else:
            text = json.dumps(input_data.text)

        if re.search(input_data.match, text, flags=flags):
            yield "positive", output
        else:
            yield "negative", output


class TextParserBlock(Block):
    class Input(BlockSchema):
        text: Any = Field(description="Text to parse")
        pattern: str = Field(description="Pattern (Regex) to parse")
        group: int = Field(description="Group number to extract", default=0)
        case_sensitive: bool = Field(description="Case sensitive match", default=True)
        dot_all: bool = Field(description="Dot matches all", default=True)

    class Output(BlockSchema):
        positive: str = Field(description="Extracted text")
        negative: str = Field(description="Original text")

    def __init__(self):
        super().__init__(
            id="3146e4fe-2cdd-4f29-bd12-0c9d5bb4deb0",
            description="This block extracts the text from the given text using the pattern (regex).",
            categories={BlockCategory.TEXT},
            input_schema=TextParserBlock.Input,
            output_schema=TextParserBlock.Output,
            test_input=[
                {"text": "Hello, World!", "pattern": "Hello, (.+)", "group": 1},
                {"text": "Hello, World!", "pattern": "Hello, (.+)", "group": 0},
                {"text": "Hello, World!", "pattern": "Hello, (.+)", "group": 2},
                {"text": "Hello, World!", "pattern": "hello,", "case_sensitive": False},
            ],
            test_output=[
                ("positive", "World!"),
                ("positive", "Hello, World!"),
                ("negative", "Hello, World!"),
                ("positive", "Hello,"),
            ],
        )

    def run(self, input_data: Input) -> BlockOutput:
        flags = 0
        if not input_data.case_sensitive:
            flags = flags | re.IGNORECASE
        if input_data.dot_all:
            flags = flags | re.DOTALL

        if isinstance(input_data.text, str):
            text = input_data.text
        else:
            text = json.dumps(input_data.text)

        match = re.search(input_data.pattern, text, flags)
        if match and input_data.group <= len(match.groups()):
            yield "positive", match.group(input_data.group)
        else:
            yield "negative", text


class TextFormatterBlock(Block):
    class Input(BlockSchema):
        texts: list[str] = Field(description="Texts (list) to format", default=[])
        named_texts: dict[str, str] = Field(
            description="Texts (dict) to format", default={}
        )
        format: str = Field(
            description="Template to format the text using `texts` and `named_texts`",
        )

    class Output(BlockSchema):
        output: str

    def __init__(self):
        super().__init__(
            id="db7d8f02-2f44-4c55-ab7a-eae0941f0c30",
            description="This block formats the given texts using the format template.",
            categories={BlockCategory.TEXT},
            input_schema=TextFormatterBlock.Input,
            output_schema=TextFormatterBlock.Output,
            test_input=[
                {"texts": ["Hello"], "format": "{texts[0]}"},
                {
                    "texts": ["Hello", "World!"],
                    "named_texts": {"name": "Alice"},
                    "format": "{texts[0]} {texts[1]} {name}",
                },
                {"format": "Hello, World!"},
            ],
            test_output=[
                ("output", "Hello"),
                ("output", "Hello World! Alice"),
                ("output", "Hello, World!"),
            ],
        )

    def run(self, input_data: Input) -> BlockOutput:
        yield "output", input_data.format.format(
            texts=input_data.texts,
            **input_data.named_texts,
        )
