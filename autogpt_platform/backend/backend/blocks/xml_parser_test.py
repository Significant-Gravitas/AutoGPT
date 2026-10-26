from unittest.mock import Mock

import pytest

from backend.blocks.xml_parser import MAX_XML_DEPTH, XMLParserBlock


def test_tokenizer_precheck_rejects_repeated_open_delimiters():
    with pytest.raises(ValueError, match="tag delimiter"):
        XMLParserBlock._normalize_tokenizer_input("<" * 40_000)


def test_tokenizer_precheck_rejects_excessive_nesting():
    xml = "<a>" * (MAX_XML_DEPTH + 1) + "x" + "</a>" * (MAX_XML_DEPTH + 1)

    with pytest.raises(ValueError, match="maximum depth"):
        XMLParserBlock._normalize_tokenizer_input(xml)


def test_tokenizer_precheck_accepts_supported_xml_shape():
    assert (
        XMLParserBlock._normalize_tokenizer_input(
            "<root><item>value</item><empty/><spaced /></root>"
        )
        == "<root><item>value</item><empty></empty><spaced></spaced></root>"
    )


def test_tokenizer_precheck_accepts_comments_with_tag_delimiters():
    XMLParserBlock._normalize_tokenizer_input(
        "<root><!-- ignored <tag> text --><item>value</item></root>"
    )


def test_tokenizer_precheck_rejects_unclosed_comment():
    with pytest.raises(ValueError, match="Unclosed XML comment"):
        XMLParserBlock._normalize_tokenizer_input("<root><!-- unfinished</root>")


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


async def _parse(xml: str):
    block = XMLParserBlock()
    return [item async for item in block.run(block.Input(input_xml=xml))]


@pytest.mark.asyncio
async def test_run_self_closing_elements_yield_empty_values():
    feed = """<feed>
  <shipment><id>A1</id><status>Delivered</status><signature/></shipment>
  <shipment><id>A2</id><status>Delivered</status><proof><signature/></proof></shipment>
</feed>"""

    assert await _parse(feed) == [
        (
            "parsed_xml",
            {
                "feed": [
                    {"shipment": {"id": "A1", "status": "Delivered", "signature": ""}},
                    {
                        "shipment": {
                            "id": "A2",
                            "status": "Delivered",
                            "proof": {"signature": ""},
                        }
                    },
                ]
            },
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "xml, expected",
    [
        ("<a><b /></a>", {"a": {"b": ""}}),
        ("<a/>", {"a": ""}),
        ("<a><b/><c>1</c></a>", {"a": {"b": "", "c": "1"}}),
    ],
)
async def test_run_self_closing_variants(xml, expected):
    assert await _parse(xml) == [("parsed_xml", expected)]


def test_tokenizer_precheck_counts_self_closing_toward_depth():
    inner = "<a>" * (MAX_XML_DEPTH - 1)
    outer = "</a>" * (MAX_XML_DEPTH - 1)
    XMLParserBlock._normalize_tokenizer_input(inner + "<b/>" + outer)

    with pytest.raises(ValueError, match="maximum depth"):
        XMLParserBlock._normalize_tokenizer_input(inner + "<a><b/></a>" + outer)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "xml",
    [
        '<feed><shipment id="A1"><status>Delivered</status></shipment></feed>',
        '<a><link href="http://x/y">t</link></a>',
        '<a><x a="1"/></a>',
        "<a><x a='1'></x></a>",
    ],
)
async def test_run_rejects_attributes_with_clear_error(xml):
    with pytest.raises(ValueError, match="XML attributes are not supported"):
        await _parse(xml)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text, expected",
    [
        ("Smith &amp; Sons", "Smith & Sons"),
        ("weight &lt; 5kg &gt; 1kg", "weight < 5kg > 1kg"),
        ("&quot;q&quot; &apos;a&apos;", "\"q\" 'a'"),
        ("&#38;&#x41;&#X41;", "&A&#X41;"),
        ("&amp;lt; stays escaped once", "&lt; stays escaped once"),
        ("&nbsp; &unknown; a & b", "&nbsp; &unknown; a & b"),
        ("&#x110000; &#xD800;", "&#x110000; &#xD800;"),
    ],
)
async def test_run_decodes_xml_entities_once(text, expected):
    assert await _parse(f"<a>{text}</a>") == [("parsed_xml", {"a": expected})]


@pytest.mark.asyncio
async def test_run_existing_shapes_unchanged():
    assert await _parse("<tag1><tag2>content</tag2></tag1>") == [
        ("parsed_xml", {"tag1": {"tag2": "content"}})
    ]
    feed = """<feed>
  <shipment><id>A1</id><dest>Ōsaka 「夜勤」</dest><status>Out for delivery 🚚</status></shipment>
  <shipment><id>A2</id><dest>Chicago, IL</dest><status>Delivered</status></shipment>
</feed>"""
    assert await _parse(feed) == [
        (
            "parsed_xml",
            {
                "feed": [
                    {
                        "shipment": {
                            "id": "A1",
                            "dest": "Ōsaka 「夜勤」",
                            "status": "Out for delivery 🚚",
                        }
                    },
                    {
                        "shipment": {
                            "id": "A2",
                            "dest": "Chicago, IL",
                            "status": "Delivered",
                        }
                    },
                ]
            },
        )
    ]
