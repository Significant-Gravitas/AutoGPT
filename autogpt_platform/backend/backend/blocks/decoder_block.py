import codecs
import re

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
        # Decode each escape sequence on its own and leave every other
        # character untouched. Running unicode_escape over the whole str
        # encodes it as UTF-8 first, turning non-ASCII text into mojibake
        # (café → cafÃ©), and re-escaping non-ASCII text collides with a
        # literal backslash in front of it.
        yield "decoded_text", _ESCAPE_SEQUENCE.sub(_decode_escape, input_data.text)


_ESCAPE_SEQUENCE = re.compile(
    r"\\(?:U[0-9a-fA-F]{8}|u[0-9a-fA-F]{4}|x[0-9a-fA-F]{2}|N\{[^}]*\}|[0-7]{1,3}|.)",
    re.DOTALL,
)


def _decode_escape(match: re.Match[str]) -> str:
    sequence = match.group(0)
    if not sequence.isascii():
        # unicode_escape keeps an unknown escape like "\日" as written.
        return sequence
    try:
        return codecs.decode(sequence, "unicode_escape")
    except UnicodeDecodeError as e:
        raise ValueError(
            f"Text contains an invalid escape sequence: {e.reason} "
            f"(at {sequence!r}, position {match.start()})"
        ) from e
