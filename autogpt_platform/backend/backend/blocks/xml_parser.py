from gravitasml.parser import Parser
from gravitasml.token import Token, tokenize

from backend.blocks._base import (
    Block,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import SchemaField

MAX_XML_SIZE = 10 * 1024 * 1024
MAX_XML_TAG_LENGTH = 4_096
MAX_XML_DEPTH = 256


class XMLParserBlock(Block):
    class Input(BlockSchemaInput):
        input_xml: str = SchemaField(description="input xml to be parsed")

    class Output(BlockSchemaOutput):
        parsed_xml: dict = SchemaField(description="output parsed xml to dict")
        error: str = SchemaField(description="Error in parsing")

    def __init__(self):
        super().__init__(
            id="286380af-9529-4b55-8be0-1d7c854abdb5",
            description="Parses XML using gravitasml to tokenize and coverts it to dict",
            input_schema=XMLParserBlock.Input,
            output_schema=XMLParserBlock.Output,
            test_input={"input_xml": "<tag1><tag2>content</tag2></tag1>"},
            test_output=[
                ("parsed_xml", {"tag1": {"tag2": "content"}}),
            ],
            effect=BlockEffect.NONE,
        )

    @staticmethod
    def _validate_tokens(tokens: list[Token]) -> None:
        """Ensure the XML has a single root element and no stray text."""
        if not tokens:
            raise ValueError("XML input is empty.")

        depth = 0
        root_seen = False

        for token in tokens:
            if token.type == "TAG_OPEN":
                if depth == 0 and root_seen:
                    raise ValueError("XML must have a single root element.")
                depth += 1
                if depth == 1:
                    root_seen = True
            elif token.type == "TAG_CLOSE":
                depth -= 1
                if depth < 0:
                    raise ValueError("Unexpected closing tag in XML input.")
            elif token.type in {"TEXT", "ESCAPE"}:
                if depth == 0 and token.value:
                    raise ValueError(
                        "XML contains text outside the root element; "
                        "wrap content in a single root tag."
                    )

        if depth != 0:
            raise ValueError("Unclosed tag detected in XML input.")
        if not root_seen:
            raise ValueError("XML must include a root element.")

    @staticmethod
    def _validate_tokenizer_input(xml: str) -> None:
        """Reject malformed or deeply nested tags in one linear pass.

        gravitasml's tokenizer searches to the end of the remaining input for
        every unmatched ``<``. Ensuring every opener has one nearby closing
        ``>`` prevents that quadratic path before the tokenizer runs.
        """
        position = 0
        depth = 0
        while True:
            tag_start = xml.find("<", position)
            if tag_start == -1:
                return

            if xml.startswith("<!--", tag_start):
                comment_end = xml.find("-->", tag_start + 4)
                if comment_end == -1:
                    raise ValueError("Unclosed XML comment.")
                position = comment_end + 3
                continue

            tag_end = xml.find(">", tag_start + 1)
            if tag_end == -1:
                raise ValueError("Unclosed tag delimiter in XML input.")
            if xml.find("<", tag_start + 1, tag_end) != -1:
                raise ValueError("Nested '<' delimiter in XML tag.")

            tag = xml[tag_start + 1 : tag_end].strip()
            if not tag:
                raise ValueError("Empty XML tag.")
            if len(tag) > MAX_XML_TAG_LENGTH:
                raise ValueError(
                    f"XML tag exceeds the {MAX_XML_TAG_LENGTH} character limit."
                )
            if tag.startswith(("!", "?")):
                raise ValueError("XML declarations and directives are not supported.")

            if tag.startswith("/"):
                depth -= 1
                if depth < 0:
                    raise ValueError("Unexpected closing tag in XML input.")
            elif not tag.endswith("/"):
                depth += 1
                if depth > MAX_XML_DEPTH:
                    raise ValueError(
                        f"XML nesting exceeds the maximum depth of {MAX_XML_DEPTH}."
                    )

            position = tag_end + 1

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        if len(input_data.input_xml) > MAX_XML_SIZE:
            raise ValueError(
                f"XML too large: {len(input_data.input_xml)} bytes > {MAX_XML_SIZE} bytes"
            )

        try:
            self._validate_tokenizer_input(input_data.input_xml)
            tokens = list(tokenize(input_data.input_xml))
            self._validate_tokens(tokens)

            parser = Parser(tokens)
            parsed_result = parser.parse()
            yield "parsed_xml", parsed_result
        except ValueError as val_e:
            raise ValueError(f"Validation error for dict:{val_e}") from val_e
        except SyntaxError as syn_e:
            # Raise as ValueError so the base Block.execute() wraps it as
            # BlockExecutionError (expected user-caused failure) instead of
            # BlockUnknownError (unexpected platform error that alerts Sentry).
            raise ValueError(f"Error in input xml syntax: {syn_e}") from syn_e
        except RecursionError as recursion_error:
            raise ValueError(
                f"XML nesting exceeds the supported parser depth of {MAX_XML_DEPTH}."
            ) from recursion_error
