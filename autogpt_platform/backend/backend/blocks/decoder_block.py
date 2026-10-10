from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockEffect,
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
            effect=BlockEffect.NONE,
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        # unicode_escape decodes bytes as Latin-1. Calling it on a str encodes
        # the str as UTF-8 first, so non-ASCII text came out as mojibake
        # (café → cafÃ©). Encode non-Latin-1 chars as backslash escapes
        # instead, which unicode_escape turns back into the original chars.
        try:
            decoded_text = input_data.text.encode("latin-1", "backslashreplace").decode(
                "unicode_escape"
            )
        except UnicodeDecodeError as e:
            raise ValueError(
                f"Text contains an invalid escape sequence: {e.reason} "
                f"(position {e.start})"
            ) from e
        yield "decoded_text", decoded_text
