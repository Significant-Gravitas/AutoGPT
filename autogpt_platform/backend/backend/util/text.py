import logging
import re

import bleach
from bleach.css_sanitizer import CSSSanitizer
from jinja2 import BaseLoader
from jinja2.exceptions import TemplateError
from jinja2.sandbox import SandboxedEnvironment
from markupsafe import Markup

logger = logging.getLogger(__name__)


def format_filter_for_jinja2(value, format_string=None):
    if format_string:
        return format_string % float(value)
    return value


class TextFormatter:
    def __init__(self, autoescape: bool = True):
        self.env = SandboxedEnvironment(loader=BaseLoader(), autoescape=autoescape)
        self.env.globals.clear()

        # Instead of clearing all filters, just remove potentially unsafe ones
        unsafe_filters = ["pprint", "tojson", "urlize", "xmlattr"]
        for f in unsafe_filters:
            if f in self.env.filters:
                del self.env.filters[f]

        self.env.filters["format"] = format_filter_for_jinja2

        # Define allowed CSS properties (sorted alphabetically, if you add more)
        allowed_css_properties = [
            "background-color",
            "border",
            "border-bottom",
            "border-color",
            "border-left",
            "border-radius",
            "border-right",
            "border-style",
            "border-top",
            "border-width",
            "bottom",
            "box-shadow",
            "clear",
            "color",
            "display",
            "float",
            "font-family",
            "font-size",
            "font-weight",
            "height",
            "left",
            "letter-spacing",
            "line-height",
            "margin-bottom",
            "margin-left",
            "margin-right",
            "margin-top",
            "padding",
            "position",
            "right",
            "text-align",
            "text-decoration",
            "text-shadow",
            "text-transform",
            "top",
            "width",
        ]

        self.css_sanitizer = CSSSanitizer(allowed_css_properties=allowed_css_properties)

        # Define allowed tags (sorted alphabetically, if you add more)
        self.allowed_tags = [
            "a",
            "b",
            "br",
            "div",
            "em",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "i",
            "img",
            "li",
            "p",
            "span",
            "strong",
            "u",
            "ul",
        ]

        # Define allowed attributes to be used on specific tags
        self.allowed_attributes = {
            "*": ["class", "style"],
            "a": ["href"],
            "img": ["src"],
        }

    def format_string(self, template_str: str, values=None, **kwargs) -> str:
        """Regular template rendering with escaping"""
        try:
            template = self.env.from_string(template_str)
            return template.render(values or {}, **kwargs)
        except TemplateError as e:
            raise ValueError(e) from e

    def format_email(
        self,
        subject_template: str,
        base_template: str,
        content_template: str,
        data=None,
        **kwargs,
    ) -> tuple[str, str]:
        """
        Special handling for email templates where content needs to be rendered as HTML
        """
        # First render the content template
        content = self.format_string(content_template, data, **kwargs)

        # Clean the HTML + CSS but don't escape it
        clean_content = bleach.clean(
            content,
            tags=self.allowed_tags,
            attributes=self.allowed_attributes,
            css_sanitizer=self.css_sanitizer,
            strip=True,
        )

        # Mark the cleaned HTML as safe using Markup
        safe_content = Markup(clean_content)

        # Render subject
        rendered_subject_template = self.format_string(subject_template, data, **kwargs)

        # Create new env just for HTML template
        html_env = SandboxedEnvironment(loader=BaseLoader(), autoescape=True)
        html_env.filters["safe"] = lambda x: (
            x if isinstance(x, Markup) else Markup(str(x))
        )

        # Render base template with the safe content
        template = html_env.from_string(base_template)
        rendered_base_template = template.render(
            data={
                "message": safe_content,
                "title": rendered_subject_template,
                "unsubscribe_link": kwargs.get("unsubscribe_link", ""),
            }
        )

        return rendered_subject_template, rendered_base_template


# ---------------------------------------------------------------------------
# CamelCase splitting
# ---------------------------------------------------------------------------

# Map of split forms back to their canonical compound terms.
# Mirrors the frontend exception list in frontend/src/lib/utils.ts.
_CAMELCASE_EXCEPTIONS: dict[str, str] = {
    "Auto GPT": "AutoGPT",
    "Open AI": "OpenAI",
    "You Tube": "YouTube",
    "Git Hub": "GitHub",
    "Linked In": "LinkedIn",
}

_CAMELCASE_EXCEPTION_RE = re.compile(
    "|".join(re.escape(k) for k in _CAMELCASE_EXCEPTIONS),
)


def split_camelcase(text: str) -> str:
    """Split CamelCase into separate words.

    Uses a single-pass character-by-character algorithm to avoid any
    regex backtracking concerns (guaranteed O(n) time).

    After splitting, known compound terms are restored via an exception
    list (e.g. ``"YouTube"`` stays ``"YouTube"`` instead of becoming
    ``"You Tube"``).  The list mirrors the frontend mapping in
    ``frontend/src/lib/utils.ts``.

    Examples::

        >>> split_camelcase("AITextGeneratorBlock")
        'AI Text Generator Block'
        >>> split_camelcase("OAuth2Block")
        'OAuth2 Block'
        >>> split_camelcase("YouTubeBlock")
        'YouTube Block'
    """
    if len(text) <= 1:
        return text

    parts: list[str] = []
    prev = 0
    for i in range(1, len(text)):
        # Insert split between lowercase/digit and uppercase: "camelCase" -> "camel|Case"
        if (text[i - 1].islower() or text[i - 1].isdigit()) and text[i].isupper():
            parts.append(text[prev:i])
            prev = i
        # Insert split between uppercase run (2+ chars) and uppercase+lowercase:
        # "AIText" -> "AI|Text".  Requires at least 3 consecutive uppercase chars
        # before the lowercase so that the left part keeps 2+ uppercase chars
        # (mirrors the original regex r"([A-Z]{2,})([A-Z][a-z])").
        elif (
            i >= 2
            and text[i - 2].isupper()
            and text[i - 1].isupper()
            and text[i].islower()
            and (i - 1 - prev) >= 2  # left part must retain at least 2 upper chars
        ):
            parts.append(text[prev : i - 1])
            prev = i - 1

    parts.append(text[prev:])
    result = " ".join(parts)

    # Restore known compound terms that should not be split.
    result = _CAMELCASE_EXCEPTION_RE.sub(
        lambda m: _CAMELCASE_EXCEPTIONS[m.group()], result
    )
    return result
