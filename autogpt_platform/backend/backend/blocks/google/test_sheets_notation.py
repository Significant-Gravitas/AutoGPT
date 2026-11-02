"""Unit tests for A1 notation helpers that do not require Google credentials."""

from backend.blocks.google.sheets import format_sheet_name, parse_a1_notation


def test_format_sheet_name_plain():
    assert format_sheet_name("Sheet1") == "Sheet1"


def test_format_sheet_name_spaces():
    assert format_sheet_name("Non-matching Leads") == "'Non-matching Leads'"


def test_format_sheet_name_apostrophe():
    assert format_sheet_name("Bob's Sheet") == "'Bob''s Sheet'"
    assert format_sheet_name("It's") == "'It''s'"


def test_format_sheet_name_starts_with_digit():
    assert format_sheet_name("2024 Report") == "'2024 Report'"


def test_parse_a1_notation_basic():
    assert parse_a1_notation("Sheet1!A1:B2") == ("Sheet1", "A1:B2")
    assert parse_a1_notation("A1:B2") == (None, "A1:B2")


def test_parse_a1_notation_quoted_with_bang():
    assert parse_a1_notation("'Q1!Plan'!A1:B2") == ("'Q1!Plan'", "A1:B2")


def test_parse_a1_notation_quoted_with_apostrophe():
    assert parse_a1_notation("'Bob''s Sheet'!A1") == ("'Bob''s Sheet'", "A1")
