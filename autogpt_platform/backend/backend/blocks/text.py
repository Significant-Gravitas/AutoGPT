import re
from typing import Any

from jinja2 import BaseLoader, Environment
from pydantic import Field

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.util import json

jinja = Environment(loader=BaseLoader())


class MatchTextPatternBlock(Block):
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
            input_schema=MatchTextPatternBlock.Input,
            output_schema=MatchTextPatternBlock.Output,
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


class ExtractTextInformationBlock(Block):
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
            input_schema=ExtractTextInformationBlock.Input,
            output_schema=ExtractTextInformationBlock.Output,
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


class FillTextTemplateBlock(Block):
    class Input(BlockSchema):
        values: dict[str, Any] = Field(description="Values (dict) to be used in format")
        format: str = Field(description="Template to format the text using `values`")

    class Output(BlockSchema):
        output: str

    def __init__(self):
        super().__init__(
            id="db7d8f02-2f44-4c55-ab7a-eae0941f0c30",
            description="This block formats the given texts using the format template.",
            categories={BlockCategory.TEXT},
            input_schema=FillTextTemplateBlock.Input,
            output_schema=FillTextTemplateBlock.Output,
            test_input=[
                {
                    "values": {"name": "Alice", "hello": "Hello", "world": "World!"},
                    "format": "{hello}, {world} {{name}}",
                },
                {
                    "values": {"list": ["Hello", " World!"]},
                    "format": "{% for item in list %}{{ item }}{% endfor %}",
                },
            ],
            test_output=[
                ("output", "Hello, World! Alice"),
                ("output", "Hello World!"),
            ],
        )

    def run(self, input_data: Input) -> BlockOutput:
        # For python.format compatibility: replace all {...} with {{..}}.
        # But avoid replacing {{...}} to {{{...}}}.
        fmt = re.sub(r"(?<!{){[ a-zA-Z0-9_]+}", r"{\g<0>}", input_data.format)
        template = jinja.from_string(fmt)
        yield "output", template.render(**input_data.values)


class CombineTextsBlock(Block):
    class Input(BlockSchema):
        input: list[str] = Field(description="text input to combine")
        delimiter: str = Field(description="Delimiter to combine texts", default="")

    class Output(BlockSchema):
        output: str = Field(description="Combined text")

    def __init__(self):
        super().__init__(
            id="e30a4d42-7b7d-4e6a-b36e-1f9b8e3b7d85",
            description="This block combines multiple input texts into a single output text.",
            categories={BlockCategory.TEXT},
            input_schema=CombineTextsBlock.Input,
            output_schema=CombineTextsBlock.Output,
            test_input=[
                {"input": ["Hello world I like ", "cake and to go for walks"]},
                {"input": ["This is a test", "Hi!"], "delimiter": "! "},
            ],
            test_output=[
                ("output", "Hello world I like cake and to go for walks"),
                ("output", "This is a test! Hi!"),
            ],
        )

    def run(self, input_data: Input) -> BlockOutput:
        combined_text = input_data.delimiter.join(input_data.input)
        yield "output", combined_text
