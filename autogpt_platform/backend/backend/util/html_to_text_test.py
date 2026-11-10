import html2text
import pytest

from backend.util.html_to_text import BoundedHTML2Text


def test_output_within_the_limit_is_what_html2text_produces():
    html = (
        "<head><title>t</title></head><body><h1>Title</h1>"
        "<ul><li>one<ul><li>two<pre>a\nb\n</pre></li></ul></li></ul>"
        "<blockquote><p>quoted&nbsp;text</p><pre>c\nd</pre></blockquote>"
        '<p>A <a href="https://example.com">link</a></p></body>'
    )
    converter = BoundedHTML2Text(max_chars=10 * len(html))

    assert converter.handle(html) == html2text.HTML2Text().handle(html)
    assert converter.truncated is False


@pytest.mark.parametrize(
    "expanding_html",
    [
        "<ul>" * 20_000 + "<pre>" + "\n" * 80_000,
        "<blockquote>" * 20_000 + "<pre>" + "\n" * 80_000,
        "<ul><li>" * 40_000,
        "<blockquote>x" * 40_000,
        "<head>" + "<ul><li>" * 40_000,  # writes nothing, takes as long
    ],
)
def test_stops_when_nesting_expands_the_output_past_the_limit(expanding_html: str):
    converter = BoundedHTML2Text(max_chars=100_000)
    converter.body_width = 0

    text = converter.handle("<p>first words</p>" + expanding_html)

    assert converter.truncated is True
    assert text.startswith("first words")
    assert len(text) <= 100_000


@pytest.mark.parametrize("tag", ["p", "pre"])
def test_keeps_the_start_of_a_text_longer_than_the_limit(tag: str):
    html = f"<{tag}>" + "some words\n" * 1_000
    converter = BoundedHTML2Text(max_chars=1_000)
    converter.body_width = 0
    unbounded = html2text.HTML2Text(bodywidth=0)

    text = converter.handle(html)

    assert converter.truncated is True
    assert 900 < len(text) <= 1_000
    assert unbounded.handle(html).startswith(text)
