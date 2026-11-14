"""Tests for Google Sheets A1 notation helpers."""

import pytest

from backend.blocks.google.sheets import format_sheet_name, parse_a1_notation


class TestFormatSheetName:
    """Tests for ``format_sheet_name``."""

    def test_plain_name_no_quotes(self) -> None:
        assert format_sheet_name("Sheet1") == "Sheet1"

    def test_name_with_space_quoted(self) -> None:
        assert format_sheet_name("Non-matching Leads") == "'Non-matching Leads'"

    def test_name_starting_with_digit_quoted(self) -> None:
        assert format_sheet_name("2024Data") == "'2024Data'"

    def test_name_with_special_char_quoted(self) -> None:
        assert format_sheet_name("Q1-Plan") == "'Q1-Plan'"

    def test_name_with_apostrophe_quoted_and_escaped(self) -> None:
        """Apostrophes must trigger quoting and be doubled inside the quotes."""
        result = format_sheet_name("Bob's Sheet")
        assert result == "'Bob''s Sheet'"

    def test_short_name_with_apostrophe(self) -> None:
        """Even a short name like 'It's' must be quoted and escaped."""
        result = format_sheet_name("It's")
        assert result == "'It''s'"

    def test_name_with_multiple_apostrophes(self) -> None:
        result = format_sheet_name("It's Bob's")
        assert result == "'It''s Bob''s'"

    def test_empty_name(self) -> None:
        assert format_sheet_name("") == ""


class TestParseA1Notation:
    """Tests for ``parse_a1_notation``."""

    def test_simple_sheet_and_range(self) -> None:
        assert parse_a1_notation("Sheet1!A1:B2") == ("Sheet1", "A1:B2")

    def test_range_only_no_sheet(self) -> None:
        assert parse_a1_notation("A1:B2") == (None, "A1:B2")

    def test_quoted_sheet_with_exclamation(self) -> None:
        """A quoted sheet name that contains '!' must not be split inside the quotes."""
        result = parse_a1_notation("'Q1!Plan'!A1:B2")
        assert result == ("'Q1!Plan'", "A1:B2")

    def test_quoted_sheet_without_exclamation(self) -> None:
        result = parse_a1_notation("'My Sheet'!A1")
        assert result == ("'My Sheet'", "A1")

    def test_plain_sheet_with_exclamation_in_range(self) -> None:
        result = parse_a1_notation("Sheet1!A1!")
        assert result == ("Sheet1", "A1!")
