"""html2text for HTML from untrusted sources."""

import html2text


class BoundedHTML2Text(html2text.HTML2Text):
    """``HTML2Text`` whose ``handle()`` stops after ``max_chars`` characters of output.

    html2text repeats the indentation of every open list and blockquote on each
    line, so its output grows with nesting depth times lines: quadratically in
    the size of a document whose author picks both. The limit therefore holds
    while converting rather than on the result; ``truncated`` says it was hit.
    """

    def __init__(self, max_chars: int) -> None:
        super().__init__()
        self._chars_left = max_chars
        self.truncated = False

    def handle(self, data: str) -> str:
        try:
            return super().handle(data)
        except _OutputLimitReached:
            self.truncated = True
            self._chars_left = 0
            self.reset()  # drop the input that was not converted
            return super().handle("")

    def o(self, data: str, puredata: bool = False, force: bool | str = False) -> None:
        # A call that writes nothing counts too: <li> makes one per open list.
        self._spend(1)
        if self.pre:
            # Preformatted lines are indented in one str.replace(), so lines that
            # cannot fit are dropped before that string is built.
            indent = max(self.blockquote, 0) + 1 + 4 * max(len(self.list), 1)
            max_newlines = self._chars_left // indent + 1
            data = "\n".join(data.split("\n", max_newlines + 1)[: max_newlines + 1])
        super().o(data, puredata, force)

    def outtextf(self, s: str) -> None:
        super().outtextf(s[: self._chars_left])
        self._spend(len(s))

    def _spend(self, chars: int) -> None:
        if self.truncated:
            return
        self._chars_left -= chars
        if self._chars_left < 0:
            raise _OutputLimitReached


class _OutputLimitReached(Exception):
    pass
