import codecs

from backend.data.block import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import SchemaField


class TextDecoderBlock(Block):
    class Input(BlockSchemaInput):
        text: str = SchemaField(
            description="A string containing escaped characters to be decoded",
            placeholder='Your entire text block with \\n and \\" escaped characters',
        )

    class Output(BlockSchemaOutput):
        decoded_text: str = SchemaField(
            description="The decoded text with escape sequences processed"
        )

    def __init__(self):
        super().__init__(
            id="2570e8fe-8447-43ed-84c7-70d657923231",
            description="Decodes a string containing escape sequences into actual text",
            categories={BlockCategory.TEXT},
            input_schema=TextDecoderBlock.Input,
            output_schema=TextDecoderBlock.Output,
            test_input={"text": """Hello\nWorld!\nThis is a \"quoted\" string."""},
            test_output=[
                (
                    "decoded_text",
                    """Hello
World!
This is a "quoted" string.""",
                )
            ],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        decoded_text = codecs.decode(input_data.text, "unicode_escape")
        yield "decoded_text", decoded_text


class TextEncoderBlock(Block):
    """
    Encodes text by adding escape sequences for special characters.
    This is the reverse operation of TextDecoderBlock.
    """

    class Input(BlockSchemaInput):
        text: str = SchemaField(
            description="A string to be encoded with escape sequences",
            placeholder="Your text with newlines and special characters",
        )

    class Output(BlockSchemaOutput):
        encoded_text: str = SchemaField(
            description="The encoded text with escape sequences added"
        )

    def __init__(self):
        super().__init__(
            id="f8e7d6c5-4b3a-2190-8765-4321fedcba98",
            description="Encodes a string by adding escape sequences for special characters",
            categories={BlockCategory.TEXT},
            input_schema=TextEncoderBlock.Input,
            output_schema=TextEncoderBlock.Output,
            test_input={
                "text": """Hello
World!
This is a "quoted" string."""
            },
            test_output=[
                (
                    "encoded_text",
                    """Hello\\nWorld!\\nThis is a "quoted" string.""",
                )
            ],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        encoded_text = codecs.encode(input_data.text, "unicode_escape").decode("ascii")
        yield "encoded_text", encoded_text
