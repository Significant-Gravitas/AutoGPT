"""Turns the body a run wrote into its email's HTML and plain-text parts.

Every format ends as the same small set of tags, each styled inline, so all
three look alike in the template and no mail client needs a stylesheet. An
image is shown only as an attachment of the email itself (``cid:``): the mail
client never fetches anything the body names.
"""

import html
import re
from collections.abc import Mapping
from enum import Enum
from html.parser import HTMLParser
from typing import Literal

import bleach
import html2text
from markdown_it import MarkdownIt
from markupsafe import Markup
from pydantic import BaseModel, ConfigDict


class BodyFormat(str, Enum):
    PLAIN = "plain"
    MARKDOWN = "markdown"
    HTML = "html"


ImageType = Literal["image/png", "image/jpeg", "image/gif", "image/webp"]
MAX_IMAGES = 8


class InlineImage(BaseModel):
    source: str
    content_type: ImageType
    data: str  # base64


class RenderedBody(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    html: Markup
    text: str


def render_body(
    body: str, body_format: BodyFormat, images: Mapping[str, str] | None = None
) -> RenderedBody:
    """``images`` maps an image source in the body to the content id of the
    attachment carrying it; any other image is shown as its alt text."""
    if body_format is BodyFormat.PLAIN:
        return RenderedBody(html=Markup(_restyle(_plain_to_html(body), {})), text=body)
    clean = _restyle(_sanitize(_to_html(body, body_format)), images or {})
    return RenderedBody(html=Markup(clean), text=_html_to_text(clean))


def image_sources(body: str, body_format: BodyFormat) -> list[str]:
    """The fetchable image sources a body names, in order, each once."""
    if body_format is BodyFormat.PLAIN:
        return []
    collector = _ImageSources()
    collector.feed(_sanitize(_to_html(body, body_format)))
    collector.close()
    return list(dict.fromkeys(collector.sources))


ALLOWED_TAGS = frozenset(
    {"a", "b", "blockquote", "br", "code", "del", "div", "em", "hr", "i", "li", "ol"}
    | {"h1", "h2", "h3", "h4", "h5", "h6", "p", "pre", "s", "span", "strong", "u"}
    | {"img", "table", "tbody", "td", "th", "thead", "tr", "ul"}
)
# No style, class or id: the template styles every tag itself.
ALLOWED_ATTRIBUTES = {
    "a": ["href", "title"],
    "img": ["src", "alt"],
    "td": ["colspan", "rowspan"],
    "th": ["colspan", "rowspan"],
}
ALLOWED_PROTOCOLS = frozenset({"http", "https", "mailto", "workspace", "data"})
LINK_SCHEMES = ("http://", "https://", "mailto:")
IMAGE_SCHEMES = ("https://", "workspace://", "data:image/")
# bleach strips a disallowed tag but keeps its text, which for these is code.
_DROPPED_WITH_CONTENT = re.compile(
    r"<(script|style|head|title|template|noscript|iframe|object|svg|math|form|button|select|textarea)\b.*?</\1\s*>",
    re.IGNORECASE | re.DOTALL,
)
_VOID_TAGS = frozenset({"br", "hr", "img"})

_BODY = "font-size:15px; line-height:24px; color:#2D2A3E;"
_STYLES = {
    "p": f"margin:0 0 16px; {_BODY}",
    "div": _BODY,
    "h1": "margin:24px 0 10px; font-size:22px; line-height:30px; font-weight:700; color:#171925;",
    "h2": "margin:22px 0 8px; font-size:19px; line-height:27px; font-weight:700; color:#171925;",
    "h3": "margin:20px 0 8px; font-size:17px; line-height:25px; font-weight:700; color:#171925;",
    "h4": "margin:18px 0 6px; font-size:15px; line-height:23px; font-weight:700; color:#171925;",
    "h5": "margin:18px 0 6px; font-size:15px; line-height:23px; font-weight:700; color:#171925;",
    "h6": "margin:18px 0 6px; font-size:15px; line-height:23px; font-weight:700; color:#575072;",
    "a": "color:#4D36AD; text-decoration:underline;",
    "ul": f"margin:0 0 16px; padding-left:22px; {_BODY}",
    "ol": f"margin:0 0 16px; padding-left:22px; {_BODY}",
    "li": "margin:0 0 6px;",
    "blockquote": "margin:0 0 16px; padding:2px 0 2px 14px; border-left:3px solid #E7E4F4; color:#575072;",
    "pre": "margin:0 0 16px; padding:12px 14px; background-color:#F5F3FC; border-radius:8px; white-space:pre-wrap; word-break:break-word; font-family:'Geist Mono', ui-monospace, Menlo, Consolas, monospace; font-size:13px; line-height:20px; color:#171925;",
    "code": "padding:1px 4px; background-color:#F5F3FC; border-radius:4px; font-family:'Geist Mono', ui-monospace, Menlo, Consolas, monospace; font-size:13px; color:#171925;",
    "hr": "margin:20px 0; border:0; border-top:1px solid #E7E4F4;",
    "img": "display:block; max-width:100%; height:auto; margin:0 0 16px; border:0; border-radius:8px;",
    "table": "margin:0 0 16px; width:100%; border-collapse:collapse; font-size:14px; line-height:20px; color:#2D2A3E;",
    "th": "padding:7px 9px; border:1px solid #E7E4F4; background-color:#F5F3FC; text-align:left; vertical-align:top; font-weight:700;",
    "td": "padding:7px 9px; border:1px solid #E7E4F4; text-align:left; vertical-align:top;",
}


def _to_html(body: str, body_format: BodyFormat) -> str:
    return _MARKDOWN.render(body) if body_format is BodyFormat.MARKDOWN else body


def _plain_to_html(body: str) -> str:
    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    return "".join(
        "<p>" + "<br>".join(html.escape(line) for line in p.splitlines()) + "</p>"
        for p in paragraphs
    )


def _sanitize(fragment: str) -> str:
    return bleach.clean(
        _DROPPED_WITH_CONTENT.sub("", fragment),
        tags=ALLOWED_TAGS,
        attributes=ALLOWED_ATTRIBUTES,
        protocols=ALLOWED_PROTOCOLS,
        strip=True,
        strip_comments=True,
    )


def _restyle(clean: str, images: Mapping[str, str]) -> str:
    """Re-serializes sanitized HTML from its parsed tags, escaping every
    attribute, so a ``>`` inside a value can't split a tag."""
    writer = _Restyler(images)
    writer.feed(clean)
    writer.close()
    return "".join(writer.out)


def _html_to_text(clean: str) -> str:
    converter = html2text.HTML2Text()
    converter.body_width = 0
    converter.images_to_alt = True
    return converter.handle(clean).strip()


class _Restyler(HTMLParser):
    def __init__(self, images: Mapping[str, str]):
        super().__init__(convert_charrefs=True)
        self.images = images
        self.out: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {name: value or "" for name, value in attrs}
        if tag == "img":
            self._image(values)
            return
        if tag == "a" and not values.get("href", "").startswith(LINK_SCHEMES):
            values.pop("href", None)
        if tag in _STYLES:
            values["style"] = _STYLES[tag]
        attributes = "".join(
            f' {name}="{html.escape(value, quote=True)}"'
            for name, value in values.items()
        )
        self.out.append(f"<{tag}{attributes}>")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if tag not in _VOID_TAGS:
            self.out.append(f"</{tag}>")

    def handle_data(self, data: str) -> None:
        self.out.append(html.escape(data, quote=False))

    def _image(self, values: dict[str, str]) -> None:
        alt = values.get("alt", "")
        content_id = self.images.get(values.get("src", ""))
        if content_id is None:
            self.out.append(html.escape(alt, quote=False))
            return
        self.out.append(
            f'<img src="cid:{html.escape(content_id, quote=True)}" '
            f'alt="{html.escape(alt, quote=True)}" style="{_STYLES["img"]}">'
        )


class _ImageSources(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.sources: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        src = dict(attrs).get("src") or ""
        if tag == "img" and src.startswith(IMAGE_SCHEMES):
            self.sources.append(src)


# Inline HTML goes through the same sanitizer as an HTML body, as on GitHub;
# breaks keep the line breaks the writer typed.
_MARKDOWN = MarkdownIt("commonmark", {"html": True, "breaks": True}).enable(
    ["table", "strikethrough"]
)
