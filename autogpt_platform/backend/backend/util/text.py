import logging

import bleach
from bleach.css_sanitizer import CSSSanitizer
from jinja2 import BaseLoader
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
        template = self.env.from_string(template_str)
        return template.render(values or {}, **kwargs)

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
