"""What the scripted model answers, read off the conversation like a model would.

The script knows two tasks. "Read handbook pages ..." asks for ``read_doc``
one page at a time, each page carrying a code word, then answers with the
words it has seen. A fact sheet followed by "second question" can be made to
overflow the upstream, so the CLI has to compact and ask again. A compaction
request is always answered with a summary that keeps every page line and the
fact the second question needs, so a resumed or retried turn can still answer.
"""

import re
from typing import Literal

from pydantic import BaseModel

PAGE_WORDS = {1: "amber-7", 2: "cobalt-14", 3: "saffron-21", 4: "teal-28"}
READ_ASK = "Read handbook pages"
NO_TOOLS = "Without calling any tools"
FACT_ASK = "second question"
COMPACTION_MARKERS = (
    "CRITICAL: Respond with TEXT ONLY",
    "detailed summary of the conversation",
)
_PAGE_LINE = re.compile(r"HANDBOOK PAGE (\d+)\. The code word is ([a-z]+-\d+)\.")
_FACT_LINE = re.compile(r"fact10 is number \d+\.")


class Decision(BaseModel):
    kind: Literal["text", "tool", "overflow"]
    text: str = ""
    page: int = 0
    compaction: bool = False


class Script:
    def __init__(
        self,
        *,
        pages: int = 2,
        overflow_chars: int | None = None,
        usage_scale: float = 1.0,
        reasoning_summary: bool = True,
        fully_cached: bool = False,
    ) -> None:
        self.pages = pages
        self.overflow_chars = overflow_chars
        self.usage_scale = usage_scale
        self.fully_cached = fully_cached
        # Whether the Responses backend summarises its reasoning; without a
        # summary a reasoning item is encrypted content alone.
        self.reasoning_summary = reasoning_summary

    def decide(self, latest: str, everything: str, size: int) -> Decision:
        """``latest`` is the newest user turn's text, tool results included."""
        if any(marker in latest for marker in COMPACTION_MARKERS):
            return Decision(kind="text", text=_summary(everything), compaction=True)
        if self.overflow_chars and FACT_ASK in latest and size > self.overflow_chars:
            return Decision(kind="overflow")
        read = {int(page) for page, _ in _PAGE_LINE.findall(everything)}
        missing = [p for p in range(1, self.pages + 1) if p not in read]
        if READ_ASK in everything and missing and NO_TOOLS not in latest:
            return Decision(kind="tool", page=missing[0])
        return Decision(kind="text", text=_answer(everything, read))

    def input_tokens(self, size: int) -> int:
        """The prompt size the model reports, scaled so a test can keep the
        CLI from compacting on its own."""
        return max(1, int(size / 4 * self.usage_scale))


def handbook_page(page: int, chars: int) -> str:
    head = f"HANDBOOK PAGE {page}. The code word is {PAGE_WORDS[page]}.\n"
    filler = f"Page {page} goes on about agents, blocks and schedules at length.\n"
    return head + filler * max(1, (chars - len(head)) // len(filler))


def fact_sheet(count: int) -> str:
    return " ".join(f"fact{i} is number {i * 7}." for i in range(count))


def _summary(everything: str) -> str:
    lines = sorted(
        {
            f"HANDBOOK PAGE {p}. The code word is {w}."
            for p, w in _PAGE_LINE.findall(everything)
        }
    )
    lines.extend(_FACT_LINE.findall(everything)[:1])
    return "<summary>\n" + "\n".join(lines or ["Nothing to keep."]) + "\n</summary>"


def _answer(everything: str, read: set[int]) -> str:
    words = ", ".join(PAGE_WORDS[page] for page in sorted(read)) or "none"
    facts = _FACT_LINE.findall(everything)
    return f"Code words: {words}." + (f" {facts[0]}" if facts else "")
