from gravitasml.parser import Parser
from gravitasml.token import Token, tokenize

from backend.blocks._base import Block, BlockOutput, BlockSchemaInput, BlockSchemaOutput
from backend.data.model import SchemaField


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
                    raise SyntaxError("Unexpected closing tag in XML input.")
            elif token.type in {"TEXT", "ESCAPE"}:
                if depth == 0 and token.value:
                    raise ValueError(
                        "XML contains text outside the root element; "
                        "wrap content in a single root tag."
                    )

        if depth != 0:
            raise SyntaxError("Unclosed tag detected in XML input.")
        if not root_seen:
            raise ValueError("XML must include a root element.")

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        # Security fix: Add size limits to prevent XML bomb attacks
        MAX_XML_SIZE = 10 * 1024 * 1024  # 10MB limit for XML input

        if len(input_data.input_xml) > MAX_XML_SIZE:
            raise ValueError(
                f"XML too large: {len(input_data.input_xml)} bytes > {MAX_XML_SIZE} bytes"
            )

        try:
            tokens = list(tokenize(input_data.input_xml))
            self._validate_tokens(tokens)

            parser = Parser(tokens)
            parsed_result = parser.parse()
            yield "parsed_xml", parsed_result
        except ValueError as val_e:
            raise ValueError(f"Validation error for dict:{val_e}") from val_e
        except SyntaxError as syn_e:
            raise SyntaxError(f"Error in input xml syntax: {syn_e}") from syn_e
