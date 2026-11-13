"""Unit tests for turning Hacker News's HTML into plain text."""

import pytest

from backend.blocks.hacker_news._html import html_to_text

# Shaped like comments from both HN APIs: entities, <p> between paragraphs, a
# line break that is only a space, indented code, and a shortened link.
COMMENT = (
    "I&#x27;d try this:<p><pre><code>    def main():\n"
    "        print(&quot;Hello, world!&quot;)\n</code></pre>\n"
    "It&#x27;s what the docs[1] suggest, more or less, if you read\n"
    "them closely.<p>[1] "
    '<a href="https:&#x2F;&#x2F;docs.example.com&#x2F;guides&#x2F;getting-started'
    '&#x2F;first-steps.html#hello" rel="nofollow">https:&#x2F;&#x2F;'
    "docs.example.com&#x2F;guides&#x2F;getting-started&#x2F;f...</a>"
)


def test_a_comment_reads_as_typed():
    assert html_to_text(COMMENT) == (
        "I'd try this:\n"
        "\n"
        "    def main():\n"
        '        print("Hello, world!")\n'
        "\n"
        "It's what the docs[1] suggest, more or less, if you read them closely.\n"
        "\n"
        "[1] https://docs.example.com/guides/getting-started/first-steps.html#hello"
    )


@pytest.mark.parametrize(
    "html, expected",
    [
        ("First.<p>Second.<p>Third.", "First.\n\nSecond.\n\nThird."),
        ("<p>Closed</p><p></p><p>paragraphs</p>", "Closed\n\nparagraphs"),
        (
            "It&#x27;s 2 &gt; 1 &amp; &quot;fine&quot; &#x2F;s",
            'It\'s 2 > 1 & "fine" /s',
        ),
        ("Not <i>that</i> one", "Not *that* one"),
        ("one line\nin the  HTML", "one line in the HTML"),
        ("line<br>break", "line\nbreak"),
        ("&lt;i&gt;typed tags&lt;/i&gt; stay text", "<i>typed tags</i> stay text"),
        ("Plain text with no HTML", "Plain text with no HTML"),
    ],
)
def test_paragraphs_entities_and_italics(html: str, expected: str):
    assert html_to_text(html) == expected


@pytest.mark.parametrize(
    "html, expected",
    [
        # HN shortens what it shows; the whole URL is in href.
        (
            '<a href="https://example.com/a/very/long/path?x=1">https://example.com/a/very/lo...</a>',
            "https://example.com/a/very/long/path?x=1",
        ),
        (
            '<a href="https://news.ycombinator.com/item?id=1">news.ycombinator.com/item?id=1</a>',
            "https://news.ycombinator.com/item?id=1",
        ),
        ('see <a href="https://x.co">our site</a>.', "see our site (https://x.co)."),
        ("<a>no href</a>", "no href"),
        ('<a href="https://x.co"></a>', "https://x.co"),
    ],
)
def test_links_keep_the_full_url(html: str, expected: str):
    assert html_to_text(html) == expected


def test_code_keeps_its_indentation_and_blank_lines():
    html = "Try:<p><pre><code>  def f():\n\n      return 1\n</code></pre>It works."
    assert html_to_text(html) == "Try:\n\n  def f():\n\n      return 1\n\nIt works."


def test_an_unclosed_code_block_is_kept():
    assert html_to_text("<pre><code>  x = 1\n  y = 2") == "  x = 1\n  y = 2"


@pytest.mark.parametrize("html", [None, "", "<p>", "<p> </p>"])
def test_nothing_to_read_gives_an_empty_string(html: str | None):
    assert html_to_text(html) == ""
