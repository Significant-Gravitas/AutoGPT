from unittest.mock import Mock

from backend.blocks.google.docs import GoogleDocsFormatTextBlock


def _make_mock_docs_service() -> Mock:
    service = Mock()
    # Ensure chained call exists: service.documents().batchUpdate(...).execute()
    service.documents.return_value.batchUpdate.return_value.execute.return_value = {}
    return service


def test_format_text_parses_shorthand_hex_color():
    block = GoogleDocsFormatTextBlock()
    service = _make_mock_docs_service()

    result = block._format_text(
        service,
        document_id="doc_1",
        start_index=1,
        end_index=2,
        bold=False,
        italic=False,
        underline=False,
        font_size=0,
        foreground_color="#FFF",
    )

    assert result["success"] is True

    # Verify request body contains correct rgbColor for white.
    _, kwargs = service.documents.return_value.batchUpdate.call_args
    requests = kwargs["body"]["requests"]
    rgb = requests[0]["updateTextStyle"]["textStyle"]["foregroundColor"]["color"][
        "rgbColor"
    ]
    assert rgb == {"red": 1.0, "green": 1.0, "blue": 1.0}


def test_format_text_parses_full_hex_color():
    block = GoogleDocsFormatTextBlock()
    service = _make_mock_docs_service()

    result = block._format_text(
        service,
        document_id="doc_1",
        start_index=1,
        end_index=2,
        bold=False,
        italic=False,
        underline=False,
        font_size=0,
        foreground_color="#FF0000",
    )

    assert result["success"] is True

    _, kwargs = service.documents.return_value.batchUpdate.call_args
    requests = kwargs["body"]["requests"]
    rgb = requests[0]["updateTextStyle"]["textStyle"]["foregroundColor"]["color"][
        "rgbColor"
    ]
    assert rgb == {"red": 1.0, "green": 0.0, "blue": 0.0}


def test_format_text_ignores_invalid_color_when_other_fields_present():
    block = GoogleDocsFormatTextBlock()
    service = _make_mock_docs_service()

    result = block._format_text(
        service,
        document_id="doc_1",
        start_index=1,
        end_index=2,
        bold=True,
        italic=False,
        underline=False,
        font_size=0,
        foreground_color="#GGG",
    )

    assert result["success"] is True
    assert "warning" in result

    # Should still apply bold, but should NOT include foregroundColor in textStyle.
    _, kwargs = service.documents.return_value.batchUpdate.call_args
    requests = kwargs["body"]["requests"]
    text_style = requests[0]["updateTextStyle"]["textStyle"]
    fields = requests[0]["updateTextStyle"]["fields"]

    assert text_style == {"bold": True}
    assert fields == "bold"


def test_format_text_invalid_color_only_does_not_call_api():
    block = GoogleDocsFormatTextBlock()
    service = _make_mock_docs_service()

    result = block._format_text(
        service,
        document_id="doc_1",
        start_index=1,
        end_index=2,
        bold=False,
        italic=False,
        underline=False,
        font_size=0,
        foreground_color="#F",
    )

    assert result["success"] is False
    assert "Invalid foreground_color" in result["message"]
    service.documents.return_value.batchUpdate.assert_not_called()
