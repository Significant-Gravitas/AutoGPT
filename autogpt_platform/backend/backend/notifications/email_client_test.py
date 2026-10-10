"""Client compatibility contracts for the shared email components."""

import re
from html.parser import HTMLParser

import pytest

from backend.notifications.template_env import create_body_environment


@pytest.fixture
def kit_html() -> str:
    return (
        create_body_environment(autoescape=True)
        .from_string(
            """{% import '_email_ui.j2' as ui with context %}
{{ ui.head_block('Client compatibility') }}
{{ ui.page_open() }}{{ ui.masthead() }}{{ ui.sheet_open() }}
{{ ui.strip_amber('Needs you') }}
{{ ui.cta('Review your plan', 'https://example.com/billing') }}
{{ ui.attention_card({'title': 'Reconnect Gmail', 'body': 'Your connection expired.',
                     'cta_label': 'Reconnect', 'cta_url': 'https://example.com/connect'}) }}
{{ ui.media_link('Watch', 'A short introduction', 'https://example.com/watch') }}
{{ ui.sheet_close() }}{{ ui.page_close() }}
{{ ui.legal_footer('Account notice', 'Preferences', 'https://example.com/settings') }}
{{ ui.internal_footer('Internal notification') }}
</td></tr></table>"""
        )
        .render(assets="https://example.com/email")
    )


def test_outlook_has_font_fallback_and_fixed_image_density(kit_html: str):
    assert "<!--[if mso]>" in kit_html
    assert "<o:PixelsPerInch>96</o:PixelsPerInch>" in kit_html
    assert re.search(
        r"body,\s*td,\s*div,\s*p,\s*a,\s*span\s*\{[^}]*font-family:\s*Arial",
        kit_html,
    )


def test_frame_and_footers_keep_fluid_width_when_styles_are_stripped(kit_html: str):
    document = EmailElements()
    document.feed(re.sub(r"<style>.*?</style>", "", kit_html, flags=re.S))
    sheets = [
        attrs
        for tag, attrs in document.elements
        if tag == "table" and "sheet" in attrs.get("class", "").split()
    ]
    assert len(sheets) == 3
    for sheet in sheets:
        assert sheet["width"] == "100%"
        assert "width:100%" in sheet["style"]
        assert "max-width:560px" in sheet["style"]
    assert len(re.findall(r'<!--\[if mso\]>.*?width="560"', kit_html, re.S)) == 3
    assert kit_html.count("<!--[if mso]></td></tr></table><![endif]-->") == 3


def test_phone_sheet_has_no_outer_side_padding_and_resets_all_corners(kit_html: str):
    document = EmailElements()
    document.feed(kit_html)
    outer_cell = next(attrs for tag, attrs in document.elements if tag == "td")
    assert "padding:32px 0 44px" in outer_cell["style"]
    assert re.search(r"\.corner\s*\{\s*border-radius:0\s*!important;", kit_html)
    rounded_sheet_elements = [
        attrs
        for _, attrs in document.elements
        if "border-radius:16px 16px 0 0" in attrs.get("style", "")
        or "border-radius:0 0 16px 16px" in attrs.get("style", "")
    ]
    assert len(rounded_sheet_elements) == 3
    assert all("corner" in attrs["class"].split() for attrs in rounded_sheet_elements)


def test_both_button_sizes_preserve_padding_in_outlook(kit_html: str):
    document = EmailElements()
    document.feed(kit_html)
    buttons = [
        attrs
        for tag, attrs in document.elements
        if tag == "td" and attrs.get("bgcolor") == "#6144DF"
    ]
    assert len(buttons) == 2
    for button, padding in zip(buttons, ("13px 28px", "8px 14px")):
        assert f"padding:{padding}" in button["style"]
        assert f"mso-padding-alt:{padding}" in button["style"]


def test_triangle_and_play_symbols_request_text_presentation(kit_html: str):
    document = EmailElements()
    document.feed(kit_html)
    assert document.text.count("▲\ufe0e") == 2
    assert document.text.count("▶\ufe0e") == 1


class EmailElements(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.elements: list[tuple[str, dict[str, str]]] = []
        self.text = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]):
        self.elements.append((tag, {key: value or "" for key, value in attrs}))

    def handle_data(self, data: str):
        self.text += data
