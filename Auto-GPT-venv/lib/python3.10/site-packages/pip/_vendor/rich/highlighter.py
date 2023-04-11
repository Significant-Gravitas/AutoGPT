from abc import ABC, abstractmethod
from typing import List, Union

from .text import Text


def _combine_regex(*regexes: str) -> str:
    """Combine a number of regexes in to a single regex.

    Returns:
        str: New regex with all regexes ORed together.
    """
    return "|".join(regexes)


class Highlighter(ABC):
    """Abstract base class for highlighters."""

    def __call__(self, text: Union[str, Text]) -> Text:
        """Highlight a str or Text instance.

        Args:
            text (Union[str, ~Text]): Text to highlight.

        Raises:
            TypeError: If not called with text or str.

        Returns:
            Text: A test instance with highlighting applied.
        """
        if isinstance(text, str):
            highlight_text = Text(text)
        elif isinstance(text, Text):
            highlight_text = text.copy()
        else:
            raise TypeError(f"str or Text instance required, not {text!r}")
        self.highlight(highlight_text)
        return highlight_text

    @abstractmethod
    def highlight(self, text: Text) -> None:
        """Apply highlighting in place to text.

        Args:
            text (~Text): A text object highlight.
        """


class NullHighlighter(Highlighter):
    """A highlighter object that doesn't highlight.

    May be used to disable highlighting entirely.

    """

    def highlight(self, text: Text) -> None:
        """Nothing to do"""


class RegexHighlighter(Highlighter):
    """Applies highlighting from a list of regular expressions."""

    highlights: List[str] = []
    base_style: str = ""

    def highlight(self, text: Text) -> None:
        """Highlight :class:`rich.text.Text` using regular expressions.

        Args:
            text (~Text): Text to highlighted.

        """

        highlight_regex = text.highlight_regex
        for re_highlight in self.highlights:
            highlight_regex(re_highlight, style_prefix=self.base_style)


class ReprHighlighter(RegexHighlighter):
    """Highlights the text typically produced from ``__repr__`` methods."""

    base_style = "repr."
    highlights = [
        r"(?P<tag_start>\<)(?P<tag_name>[\w\-\.\:]*)(?P<tag_contents>[\w\W]*?)(?P<tag_end>\>)",
        r"(?P<attrib_name>[\w_]{1,50})=(?P<attrib_value>\"?[\w_]+\"?)?",
        r"(?P<brace>[\{\[\(\)\]\}])",
        _combine_regex(
            r"(?P<ipv4>[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3})",
            r"(?P<ipv6>([A-Fa-f0-9]{1,4}::?){1,7}[A-Fa-f0-9]{1,4})",
            r"(?P<eui64>(?:[0-9A-Fa-f]{1,2}-){7}[0-9A-Fa-f]{1,2}|(?:[0-9A-Fa-f]{1,2}:){7}[0-9A-Fa-f]{1,2}|(?:[0-9A-Fa-f]{4}\.){3}[0-9A-Fa-f]{4})",
            r"(?P<eui48>(?:[0-9A-Fa-f]{1,2}-){5}[0-9A-Fa-f]{1,2}|(?:[0-9A-Fa-f]{1,2}:){5}[0-9A-Fa-f]{1,2}|(?:[0-9A-Fa-f]{4}\.){2}[0-9A-Fa-f]{4})",
            r"(?P<call>[\w\.]*?)\(",
            r"\b(?P<bool_true>True)\b|\b(?P<bool_false>False)\b|\b(?P<none>None)\b",
            r"(?P<ellipsis>\.\.\.)",
            r"(?P<number>(?<!\w)\-?[0-9]+\.?[0-9]*(e[\-\+]?\d+?)?\b|0x[0-9a-fA-F]*)",
            r"(?P<path>\B(\/[\w\.\-\_\+]+)*\/)(?P<filename>[\w\.\-\_\+]*)?",
            r"(?<![\\\w])(?P<str>b?\'\'\'.*?(?<!\\)\'\'\'|b?\'.*?(?<!\\)\'|b?\"\"\".*?(?<!\\)\"\"\"|b?\".*?(?<!\\)\")",
            r"(?P<uuid>[a-fA-F0-9]{8}\-[a-fA-F0-9]{4}\-[a-fA-F0-9]{4}\-[a-fA-F0-9]{4}\-[a-fA-F0-9]{12})",
            r"(?P<url>(file|https|http|ws|wss):\/\/[0-9a-zA-Z\$\-\_\+\!`\(\)\,\.\?\/\;\:\&\=\%\#]*)",
        ),
    ]


class JSONHighlighter(RegexHighlighter):
    """Highlights JSON"""

    base_style = "json."
    highlights = [
        _combine_regex(
            r"(?P<brace>[\{\[\(\)\]\}])",
            r"\b(?P<bool_true>true)\b|\b(?P<bool_false>false)\b|\b(?P<null>null)\b",
            r"(?P<number>(?<!\w)\-?[0-9]+\.?[0-9]*(e[\-\+]?\d+?)?\b|0x[0-9a-fA-F]*)",
            r"(?<![\\\w])(?P<str>b?\".*?(?<!\\)\")",
        ),
        r"(?<![\\\w])(?P<key>b?\".*?(?<!\\)\")\:",
    ]


if __name__ == "__main__":  # pragma: no cover
    from .console import Console

    console = Console()
    console.print("[bold green]hello world![/bold green]")
    console.print("'[bold green]hello world![/bold green]'")

    console.print(" /foo")
    console.print("/foo/")
    console.print("/foo/bar")
    console.print("foo/bar/baz")

    console.print("/foo/bar/baz?foo=bar+egg&egg=baz")
    console.print("/foo/bar/baz/")
    console.print("/foo/bar/baz/egg")
    console.print("/foo/bar/baz/egg.py")
    console.print("/foo/bar/baz/egg.py word")
    console.print(" /foo/bar/baz/egg.py word")
    console.print("foo /foo/bar/baz/egg.py word")
    console.print("foo /foo/bar/ba._++z/egg+.py word")
    console.print("https://example.org?foo=bar#header")

    console.print(1234567.34)
    console.print(1 / 2)
    console.print(-1 / 123123123123)

    console.print(
        "127.0.1.1 bar 192.168.1.4 2001:0db8:85a3:0000:0000:8a2e:0370:7334 foo"
    )
