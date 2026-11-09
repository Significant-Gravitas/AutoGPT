"""Tests for csv_delimiter — guessing a CSV/TSV delimiter by parsing."""

import time

import pytest

from backend.util.csv_delimiter import _lines, guess_delimiter
from backend.util.file_content_parser import parse_file_content


def test_semicolon_file_declared_csv_parses_into_rows():
    content = "Name;Score\nAlice;90\nBob;85"
    result = parse_file_content(content, "csv")
    assert result == [["Name", "Score"], ["Alice", "90"], ["Bob", "85"]]


def test_pipe_file_declared_tsv_parses_into_rows():
    content = "Name|Score\nAlice|90"
    result = parse_file_content(content, "tsv")
    assert result == [["Name", "Score"], ["Alice", "90"]]


def test_colon_file_declared_csv_parses_into_rows():
    result = parse_file_content("Name:Score\nAlice:90", "csv")
    assert result == [["Name", "Score"], ["Alice", "90"]]


def test_quoted_field_spanning_lines_stays_one_field():
    content = 'a;"multi\nline";c\n1;2;3'
    result = parse_file_content(content, "csv")
    assert result == [["a", "multi\nline", "c"], ["1", "2", "3"]]


def test_lines_split_on_newline_only_and_keep_their_endings():
    assert list(_lines("a\nb\r\nc\rd")) == ["a\n", "b\r\n", "c\rd"]
    assert list(_lines("")) == []
    assert list(_lines("end\n")) == ["end\n"]


def test_declared_delimiter_is_not_a_candidate():
    assert guess_delimiter("a,b\n1,2", ",") is None


def test_most_consistent_delimiter_wins():
    # Both "," and ";" split the first row in two; only ";" splits the rest.
    assert guess_delimiter("a;b,c\n1;2\n3;4", "\t") == ";"


def test_punctuation_only_in_a_single_column_header_stays_text():
    """A delimiter that splits only the header, not the rows, isn't one."""
    content = "label;note\nX\nY"
    assert guess_delimiter(content, ",") is None
    assert parse_file_content(content, "csv") == content


def test_header_only_file_with_fallback_delimiter():
    assert parse_file_content("Name;Score", "csv") == [["Name", "Score"]]


def test_one_ragged_row_in_ten_is_tolerated():
    content = "a;b\n" + "1;2\n" * 9 + "3;4;5"
    result = parse_file_content(content, "csv")
    assert result == [["a", "b"]] + [["1", "2"]] * 9 + [["3", "4", "5"]]


def test_every_complete_row_counts():
    # Nine two-field rows, then three one-field rows: 9 in 12 agree.
    content = "a;b\n" * 9 + "bad\n" + "x" * 8151 + "\nmore\n"
    assert guess_delimiter(content, ",") is None


def test_first_row_longer_than_8_kb_parses():
    content = "x" * 8193 + ";Score\nAlice;90\nBob;85"
    result = parse_file_content(content, "csv")
    assert result == [["x" * 8193, "Score"], ["Alice", "90"], ["Bob", "85"]]


def test_blank_lines_do_not_count_toward_the_sample():
    content = "label;note\n" + "\n" * 49 + "X\nY"
    assert guess_delimiter(content, ",") is None


def test_candidate_csv_reader_cannot_read_is_skipped():
    # A field over csv.field_size_limit() makes the reader raise csv.Error.
    assert guess_delimiter("a;" + "x" * 200_000, ",") is None


@pytest.mark.parametrize("delimiter_in_sample", ["", ";"])
def test_quote_heavy_single_column_parses_quickly(delimiter_in_sample):
    """csv.Sniffer ran for over a minute on this 1 KB shape before CPython
    3.13.15 (gh-109638), and for hours at 8 KB; parsing it now takes
    milliseconds, so the bound only has to leave room for a CI pause."""
    quotes = '"' * 480
    content = f':\n{quotes}aa: {delimiter_in_sample}{quotes}a =d-] "f"\n[\r\n= "'
    start = time.perf_counter()
    result = parse_file_content(content, "csv")
    assert time.perf_counter() - start < 10.0
    assert result == content
