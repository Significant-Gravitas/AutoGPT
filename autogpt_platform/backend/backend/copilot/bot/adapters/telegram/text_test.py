"""Tests for CommonMark → Telegram HTML conversion."""

from .text import to_html


def test_bold_and_links_become_html_tags():
    assert to_html("**hi** [docs](https://x.io)") == (
        '<b>hi</b> <a href="https://x.io">docs</a>'
    )


def test_inline_code_and_code_blocks():
    assert to_html("run `ls` now") == "run <code>ls</code> now"
    assert to_html("```py\nprint(1)\n```") == "<pre>print(1)</pre>"


def test_html_in_input_is_escaped_not_rendered():
    # Model/user output must not be able to inject live tags.
    assert to_html("<script>alert(1)</script>") == (
        "&lt;script&gt;alert(1)&lt;/script&gt;"
    )


def test_plain_text_survives_unchanged():
    assert to_html("just words, no markup") == "just words, no markup"


def test_markdown_inside_code_spans_stays_literal():
    block = "```" + "\n**kwargs** and [x](y)\n" + "```"
    assert to_html(block) == "<pre>**kwargs** and [x](y)</pre>"
    assert to_html("`**bold**`") == "<code>**bold**</code>"


def test_double_quotes_escape_so_hrefs_stay_valid():
    assert to_html('a "quoted" word') == "a &quot;quoted&quot; word"
