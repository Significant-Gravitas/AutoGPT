"""Reading a worker's own words: the one line a card shows from its last
message, and whether a result it gave is really a failure.

Pure text helpers, so ``contract`` can use them without a cycle.
"""

import re

RESULT_LINE_CHARS = 140

_FENCE = re.compile(r"```.*?```", re.DOTALL)
_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_CODE = re.compile(r"`([^`]*)`")
_EMPHASIS = re.compile(r"(\*\*|__|\*|~~)(?=\S)(.+?)(?<=\S)\1")
_LINE_MARK = re.compile(r"^\s{0,3}(#{1,6}\s+|>\s?|[-*+]\s+|\d+[.)]\s+)")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s")
_RULE = re.compile(r"^\s*([-*_=|:]\s*){3,}$")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

# A result that says the work wasn't done: the thing to work on was missing,
# or the worker couldn't get at it.
_FAILURE = re.compile(
    r"\b(couldn'?t|could not|can'?t|cannot|unable to|wasn'?t able to|"
    r"was not able to|failed to|didn'?t|did not)\s+(find|locate|access|open|"
    r"read|get|load|reach|download|fetch|see|complete|finish|do|summari[sz]e)"
    r"\b(?!\s+(any\s+)?(issues?|problems?|errors?|bugs?|mistakes?|typos?)\b)"
    r"|\bno\s+(\w+\s+){0,3}(pdf|file|document|attachment|report|spreadsheet|"
    r"image|link|data|input)s?\s+(was|were|is|are)\s+(attached|provided|"
    r"uploaded|shared|found|available)\b"
    r"|\b(wasn'?t|weren'?t|isn'?t|aren'?t|was not|were not|is not|are not)\s+"
    r"(attached|provided|uploaded|shared|found|available)\b"
    r"|\b(file|pdf|attachment|document|input)s?\s+(is|are|was|were)\s+missing\b"
    r"|\bmissing\s+(the\s+|a\s+|an\s+)?(\w+\s+){0,2}(file|pdf|attachment|"
    r"document|input)s?\b",
    re.IGNORECASE,
)


def plain(text: str | None, headings: bool = True) -> list[str]:
    """The text's lines without markdown, empty ones and rules dropped (and
    headings, unless *headings*)."""
    body = _FENCE.sub(" ", text or "")
    lines = []
    for raw in body.splitlines():
        if not headings and _HEADING.match(raw):
            continue
        line = _LINE_MARK.sub("", raw)
        if _RULE.match(line):
            continue
        line = _IMAGE.sub(r"\1", line)
        line = _LINK.sub(r"\1", line)
        line = _CODE.sub(r"\1", line)
        line = _EMPHASIS.sub(r"\2", line)
        line = " ".join(line.split())
        if line:
            lines.append(line)
    return lines


def result_line(text: str | None, limit: int = RESULT_LINE_CHARS) -> str:
    """The first sentence of the first line, as a card shows a result; a
    heading counts only when there is nothing else."""
    lines = plain(text, headings=False) or plain(text)
    if not lines:
        return ""
    first = _SENTENCE_END.split(lines[0], maxsplit=1)[0]
    return trim(first, limit)


def ask_line(text: str | None, limit: int = RESULT_LINE_CHARS) -> str:
    """The question a message ends on, else its first sentence."""
    sentences = [s for line in plain(text) for s in _SENTENCE_END.split(line)]
    question = next((s for s in reversed(sentences) if s.endswith("?")), None)
    return trim(question, limit) if question else result_line(text, limit)


def reads_as_failure(text: str | None) -> bool:
    """Whether a result says the work couldn't be done."""
    return bool(_FAILURE.search(" ".join(plain(text))))


def trim(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    cut = flat[: limit - 1]
    space = cut.rfind(" ")
    if space > limit // 2:
        cut = cut[:space]
    return cut.rstrip(" ,;:-") + "…"
