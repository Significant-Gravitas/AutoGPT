"""Shared Jinja configuration for HTML and plain-text email bodies."""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

TEMPLATE_DIR = Path(__file__).parent / "templates"


def create_body_environment(*, autoescape: bool) -> Environment:
    return Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=autoescape,
        trim_blocks=True,
        lstrip_blocks=True,
    )
