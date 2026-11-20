"""Plain text from the HTML Hacker News stores for comments, posts and profiles.

HN keeps what people type as a little HTML: <p> between paragraphs, <i> for
text typed between asterisks, <pre><code> for indented code, and an <a> for
every URL, whose visible text HN cuts short ("https://example.com/a/lo...")
while href keeps the whole URL. This turns it back into the text the person
typed: blank lines between paragraphs, *asterisks* for italics, code kept
exactly as written, and full URLs.
"""

import re
from html.parser import HTMLParser

_SPACES = re.compile(r"\s+")
_SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*:(//)?", re.IGNORECASE)


def html_to_text(html: str | None) -> str:
    """The plain text of HN's HTML, or an empty string when there is none."""
    if not html:
        return ""
    parser = _HackerNewsHTML()
    parser.feed(html)
    parser.close()
    return parser.text()


class _HackerNewsHTML(HTMLParser):
    def __init__(self) -> None:
        # Entities such as &#x27; and &gt; arrive in handle_data decoded.
        super().__init__(convert_charrefs=True)
        self._blocks: list[str] = []
        self._paragraph: list[str] = []
        self._code: list[str] | None = None  # set while inside <pre>
        self._link: list[str] | None = None  # set while inside <a>
        self._href = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "p":
            self._end_paragraph()
        elif tag == "pre":
            self._end_paragraph()
            self._code = []
        elif tag == "br":
            self._write("\n")
        elif tag in ("i", "em"):
            self._write("*")
        elif tag == "a":
            self._link = []
            self._href = dict(attrs).get("href") or ""

    def handle_endtag(self, tag: str) -> None:
        if tag == "p":
            self._end_paragraph()
        elif tag == "pre":
            self._end_code()
        elif tag in ("i", "em"):
            self._write("*")
        elif tag == "a" and self._link is not None:
            visible = "".join(self._link)
            self._link = None
            self._write(_link_text(visible, self._href))

    def handle_data(self, data: str) -> None:
        if self._link is not None:
            self._link.append(data)
        elif self._code is not None:
            self._code.append(data)
        else:
            # Line breaks in HN's HTML are only spaces; <br> adds a real one.
            self._paragraph.append(_SPACES.sub(" ", data))

    def text(self) -> str:
        self._end_code()
        self._end_paragraph()
        return "\n\n".join(self._blocks)

    def _write(self, text: str) -> None:
        if self._code is not None:
            self._code.append(text)
        else:
            self._paragraph.append(text)

    def _end_paragraph(self) -> None:
        lines = "".join(self._paragraph).split("\n")
        self._paragraph = []
        paragraph = "\n".join(" ".join(line.split()) for line in lines).strip()
        if paragraph:
            self._blocks.append(paragraph)

    def _end_code(self) -> None:
        if self._code is None:
            return
        # Keep the indentation: HN only shows text indented by 2+ spaces as code.
        code = "".join(self._code).strip("\n").rstrip()
        self._code = None
        if code:
            self._blocks.append(code)


def _link_text(visible: str, href: str) -> str:
    """The full URL for one of HN's links, which show it shortened with '...'."""
    visible = visible.strip()
    shown = visible.removesuffix("...")
    if not href:
        return visible
    if not shown or _SCHEME.sub("", href).startswith(_SCHEME.sub("", shown)):
        return href
    return f"{visible} ({href})"
