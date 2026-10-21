"""Turns the body a run wrote into its email's HTML and plain-text parts.

Every format ends as the same small set of tags, each styled inline, so all
three look alike in the template and no mail client needs a stylesheet.
"""

import html
import re
from enum import Enum

import bleach
import html2text
from markdown_it import MarkdownIt
from markupsafe import Markup
from pydantic import BaseModel, ConfigDict


class BodyFormat(str, Enum):
    PLAIN = "plain"
    MARKDOWN = "markdown"
    HTML = "html"


class RenderedBody(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    html: Markup
    text: str


def render_body(body: str, body_format: BodyFormat) -> RenderedBody:
    if body_format is BodyFormat.PLAIN:
        clean = _plain_to_html(body)
        return RenderedBody(html=Markup(_style(clean)), text=body)
    fragment = _MARKDOWN.render(body) if body_format is BodyFormat.MARKDOWN else body
    clean = _sanitize(fragment)
    return RenderedBody(html=Markup(_style(clean)), text=_html_to_text(clean))


# No images: any remote image in mail is a tracking pixel. No style, class or
# id: the template styles every tag itself.
ALLOWED_TAGS = frozenset(
    {"a", "b", "blockquote", "br", "code", "del", "div", "em", "hr", "i", "li", "ol"}
    | {"h1", "h2", "h3", "h4", "h5", "h6", "p", "pre", "s", "span", "strong", "u"}
    | {"table", "tbody", "td", "th", "thead", "tr", "ul"}
)
ALLOWED_ATTRIBUTES = {
    "a": ["href", "title"],
    "td": ["colspan", "rowspan"],
    "th": ["colspan", "rowspan"],
}
ALLOWED_PROTOCOLS = frozenset({"http", "https", "mailto"})
# bleach strips a disallowed tag but keeps its text, which for these is code.
_DROPPED_WITH_CONTENT = re.compile(
    r"<(script|style|head|title|template|noscript|iframe|object|svg|math|form|button|select|textarea)\b.*?</\1\s*>",
    re.IGNORECASE | re.DOTALL,
)

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
    "table": "margin:0 0 16px; width:100%; border-collapse:collapse; font-size:14px; line-height:20px; color:#2D2A3E;",
    "th": "padding:7px 9px; border:1px solid #E7E4F4; background-color:#F5F3FC; text-align:left; vertical-align:top; font-weight:700;",
    "td": "padding:7px 9px; border:1px solid #E7E4F4; text-align:left; vertical-align:top;",
}
_OPENING_TAG = re.compile(rf"<({'|'.join(_STYLES)})\b([^>]*)>")


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


def _style(clean: str) -> str:
    """Safe only on sanitized HTML, which carries no style attribute of its own."""
    return _OPENING_TAG.sub(
        lambda m: f'<{m.group(1)}{m.group(2).rstrip(" /")} style="{_STYLES[m.group(1)]}">',
        clean,
    )


def _html_to_text(clean: str) -> str:
    converter = html2text.HTML2Text()
    converter.body_width = 0
    converter.ignore_images = True
    return converter.handle(clean).strip()


def _image_as_link(self, tokens, idx, options, env) -> str:
    token = tokens[idx]
    src = html.escape(str(token.attrGet("src") or ""), quote=True)
    label = html.escape(token.content or "image")
    return f'<a href="{src}">{label}</a>'


# Inline HTML goes through the same sanitizer as an HTML body, as on GitHub;
# breaks keep the line breaks the writer typed.
_MARKDOWN = MarkdownIt("commonmark", {"html": True, "breaks": True}).enable(
    ["table", "strikethrough"]
)
_MARKDOWN.add_render_rule("image", _image_as_link)
