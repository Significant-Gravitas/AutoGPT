import re

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

# gravitasml does not decode entities; decode the XML predefined ones and
# numeric references once. Unknown entities and bare "&" stay as-is.
_XML_ENTITY_RE = re.compile(
    r"&(?:#x([0-9a-fA-F]{1,6})|#([0-9]{1,7})|(amp|lt|gt|quot|apos));"
)
_XML_NAMED_ENTITIES = {"amp": "&", "lt": "<", "gt": ">", "quot": '"', "apos": "'"}


def _is_xml_char(codepoint: int) -> bool:
    """XML 1.0 Char production."""
    return (
        codepoint in (0x9, 0xA, 0xD)
        or 0x20 <= codepoint <= 0xD7FF
        or 0xE000 <= codepoint <= 0xFFFD
        or 0x10000 <= codepoint <= 0x10FFFF
    )


def _decode_xml_entity(match: re.Match[str]) -> str:
    hex_code, dec_code, name = match.groups()
    if name:
        return _XML_NAMED_ENTITIES[name]
    codepoint = int(hex_code, 16) if hex_code else int(dec_code)
    if not _is_xml_char(codepoint):
        return match.group(0)
    return chr(codepoint)


def _protect_backslashes(xml: str) -> tuple[str, str | None]:
    """Swap backslashes for a private-use character absent from the input.

    gravitasml tokenizes a backslash at the start of a text run plus the next
    character as an ESCAPE token, which its parser drops. The caller restores
    the backslashes in the tokens.
    """
    if "\\" not in xml:
        return xml, None
    used = set(xml)
    sentinel = next(
        (chr(cp) for cp in range(0xE000, 0xF900) if chr(cp) not in used), None
    )
    if sentinel is None:
        raise ValueError("XML input uses every private-use character.")
    return xml.replace("\\", sentinel), sentinel


def _mixed_content_error(tag: str) -> str:
    return (
        f"Mixed text and child elements are not supported: <{tag}> has both. "
        "Wrap the text in its own tag."
    )


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
        # (tag, has_text, has_child) per open element. gravitasml drops text on
        # elements that have children, so reject the mix. Whitespace-only text
        # never becomes a token, so indentation is unaffected.
        open_elements: list[tuple[str, bool, bool]] = []

        for token in tokens:
            if token.type == "TAG_OPEN":
                if depth == 0 and root_seen:
                    raise ValueError("XML must have a single root element.")
                if open_elements:
                    parent, has_text, _ = open_elements[-1]
                    if has_text:
                        raise ValueError(_mixed_content_error(parent))
                    open_elements[-1] = (parent, has_text, True)
                open_elements.append((token.value, False, False))
                depth += 1
                if depth == 1:
                    root_seen = True
            elif token.type == "TAG_CLOSE":
                if open_elements:
                    open_elements.pop()
                depth -= 1
                if depth < 0:
                    raise ValueError("Unexpected closing tag in XML input.")
            elif token.type in {"TEXT", "ESCAPE"}:
                if depth == 0 and token.value:
                    raise ValueError(
                        "XML contains text outside the root element; "
                        "wrap content in a single root tag."
                    )
                if open_elements and token.value:
                    tag, _, has_child = open_elements[-1]
                    if has_child:
                        raise ValueError(_mixed_content_error(tag))
                    open_elements[-1] = (tag, True, has_child)

        if depth != 0:
            raise ValueError("Unclosed tag detected in XML input.")
        if not root_seen:
            raise ValueError("XML must include a root element.")

    @staticmethod
    def _normalize_tokenizer_input(xml: str) -> str:
        """Reject malformed or deeply nested tags in one linear pass and return
        the input rewritten for gravitasml.

        gravitasml's tokenizer searches to the end of the remaining input for
        every unmatched ``<``. Ensuring every opener has one nearby closing
        ``>`` prevents that quadratic path before the tokenizer runs.

        gravitasml has no self-closing tags or attributes: ``<x/>`` is turned
        into text and attributes end up in the tag name. Self-closing tags are
        rewritten to ``<x></x>``; attributes are rejected.
        """
        pieces: list[str] = []
        position = 0
        depth = 0
        while True:
            tag_start = xml.find("<", position)
            if tag_start == -1:
                pieces.append(xml[position:])
                return "".join(pieces)

            if xml.startswith("<!--", tag_start):
                comment_end = xml.find("-->", tag_start + 4)
                if comment_end == -1:
                    raise ValueError("Unclosed XML comment.")
                pieces.append(xml[position : comment_end + 3])
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
            if "=" in tag or '"' in tag or "'" in tag:
                raise ValueError(f"XML attributes are not supported: <{tag}>")

            pieces.append(xml[position:tag_start])
            if tag.startswith("/"):
                depth -= 1
                if depth < 0:
                    raise ValueError("Unexpected closing tag in XML input.")
                pieces.append(xml[tag_start : tag_end + 1])
            elif tag.endswith("/"):
                if depth + 1 > MAX_XML_DEPTH:
                    raise ValueError(
                        f"XML nesting exceeds the maximum depth of {MAX_XML_DEPTH}."
                    )
                name = tag[:-1].strip()
                pieces.append(f"<{name}></{name}>")
            else:
                depth += 1
                if depth > MAX_XML_DEPTH:
                    raise ValueError(
                        f"XML nesting exceeds the maximum depth of {MAX_XML_DEPTH}."
                    )
                pieces.append(xml[tag_start : tag_end + 1])

            position = tag_end + 1

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        if len(input_data.input_xml) > MAX_XML_SIZE:
            raise ValueError(
                f"XML too large: {len(input_data.input_xml)} bytes > {MAX_XML_SIZE} bytes"
            )

        try:
            xml = self._normalize_tokenizer_input(input_data.input_xml)
            xml, sentinel = _protect_backslashes(xml)
            tokens = list(tokenize(xml))
            if sentinel:
                # Restore before entity decoding: "&#xE000;" must not become "\\".
                for token in tokens:
                    token.value = token.value.replace(sentinel, "\\")
            self._validate_tokens(tokens)
            for token in tokens:
                if token.type == "TEXT":
                    token.value = _XML_ENTITY_RE.sub(_decode_xml_entity, token.value)

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
