"""Guess a CSV/TSV delimiter by parsing, for a file whose declared delimiter
leaves its first row a single column (a tab-separated file named .csv, say).

csv.Sniffer is not used: before CPython 3.13.15 its regular expressions
backtrack catastrophically on quote-heavy text (gh-109638, gh-98820), and `re`
holds the GIL while it matches, so a few KB of crafted CSV stopped the whole
process. csv.reader takes time in proportion to what it reads.
"""

import csv
import itertools
from collections.abc import Iterator

# In order of preference when two fit equally well. The Sniffer also tried a
# space, which splits ordinary prose into columns, so it isn't a candidate.
_FALLBACK_DELIMITERS = (",", "\t", ";", "|", ":")
# Non-blank rows read per candidate. Like the Sniffer, a delimiter is only
# taken when at least nine in ten of them agree on the number of fields.
_SAMPLE_ROWS = 50


def guess_delimiter(content: str, declared: str) -> str | None:
    """The fallback delimiter that splits the first row into at least two
    fields and at least nine in ten of the first non-blank rows into that
    many, or None."""
    best: tuple[float, int] = (0.0, 0)
    guessed = None
    for candidate in _FALLBACK_DELIMITERS:
        if candidate == declared or candidate not in content:
            continue
        rows = _sample_rows(content, candidate)
        if not rows or len(rows[0]) < 2:
            continue
        if candidate == ":" and _splits_urls_or_paths(rows):
            continue
        width = len(rows[0])
        consistent = sum(len(row) == width for row in rows)
        if consistent * 10 < len(rows) * 9:
            continue
        score = (consistent / len(rows), width)
        if score > best:
            best, guessed = score, candidate
    return guessed


def _sample_rows(content: str, delimiter: str) -> list[list[str]]:
    """The first non-blank rows `delimiter` splits the content into, or none
    when csv.reader can't read it that way (a field over its size limit)."""
    reader = csv.reader(_lines(content), delimiter=delimiter)
    try:
        return list(itertools.islice((row for row in reader if any(row)), _SAMPLE_ROWS))
    except csv.Error:
        return []


def _splits_urls_or_paths(rows: list[list[str]]) -> bool:
    """Whether splitting on ":" cut a URL scheme or a drive letter off a value
    (`https://…`, `C:\\…`) rather than separating columns."""
    return any(field.startswith(("/", "\\")) for row in rows for field in row[1:])


def _lines(content: str) -> Iterator[str]:
    """The content's lines with their endings, split on "\\n" as io.StringIO
    splits them, but sliced off one at a time, so reading the first rows
    doesn't copy the whole text (StringIO keeps four bytes per character).
    csv.reader joins a quoted field's lines back together itself."""
    start = 0
    while start < len(content):
        end = content.find("\n", start) + 1 or len(content)
        yield content[start:end]
        start = end
