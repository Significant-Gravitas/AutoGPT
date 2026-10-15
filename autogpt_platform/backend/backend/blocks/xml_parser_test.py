from unittest.mock import Mock

import pytest

from backend.blocks.xml_parser import MAX_XML_DEPTH, XMLParserBlock


def test_tokenizer_precheck_rejects_repeated_open_delimiters():
    with pytest.raises(ValueError, match="tag delimiter"):
        XMLParserBlock._validate_tokenizer_input("<" * 40_000)


def test_tokenizer_precheck_rejects_excessive_nesting():
    xml = "<a>" * (MAX_XML_DEPTH + 1) + "x" + "</a>" * (MAX_XML_DEPTH + 1)

    with pytest.raises(ValueError, match="maximum depth"):
        XMLParserBlock._validate_tokenizer_input(xml)


def test_tokenizer_precheck_accepts_supported_xml_shape():
    XMLParserBlock._validate_tokenizer_input(
        '<root><item id="one">value</item><empty/></root>'
    )


def test_tokenizer_precheck_accepts_comments_with_tag_delimiters():
    XMLParserBlock._validate_tokenizer_input(
        "<root><!-- ignored <tag> text --><item>value</item></root>"
    )


def test_tokenizer_precheck_rejects_unclosed_comment():
    with pytest.raises(ValueError, match="Unclosed XML comment"):
        XMLParserBlock._validate_tokenizer_input("<root><!-- unfinished</root>")


@pytest.mark.asyncio
async def test_run_rejects_pathological_input_before_tokenizing(mocker):
    block = XMLParserBlock()
    tokenize_mock = mocker.patch(
        "backend.blocks.xml_parser.tokenize",
        Mock(side_effect=AssertionError("tokenizer must not run")),
    )
    input_data = block.Input(input_xml="<" * 40_000)

    with pytest.raises(ValueError, match="tag delimiter"):
        _ = [item async for item in block.run(input_data)]

    tokenize_mock.assert_not_called()


@pytest.mark.asyncio
async def test_run_preserves_supported_xml_comments():
    block = XMLParserBlock()
    input_data = block.Input(
        input_xml="<root><!-- ignored <tag> text --><item>value</item></root>"
    )

    outputs = [item async for item in block.run(input_data)]

    assert outputs == [("parsed_xml", {"root": {"item": "value"}})]


@pytest.mark.asyncio
async def test_run_maps_parser_recursion_to_input_error(mocker):
    block = XMLParserBlock()
    mocker.patch(
        "backend.blocks.xml_parser.tokenize",
        return_value=[],
    )
    mocker.patch.object(block, "_validate_tokens", side_effect=RecursionError)

    with pytest.raises(ValueError, match="supported parser depth"):
        _ = [item async for item in block.run(block.Input(input_xml="<a></a>"))]
