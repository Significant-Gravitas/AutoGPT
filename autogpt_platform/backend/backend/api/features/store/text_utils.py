"""Small text helpers shared across store search modules."""

import re

_MAX_CAMELCASE_INPUT_LEN = 500


def split_camelcase(text: str) -> str:
    """Split CamelCase into separate words.

    Both regex patterns are linear-time (no nested quantifiers).

    Examples::

        >>> split_camelcase("AITextGeneratorBlock")
        'AI Text Generator Block'
        >>> split_camelcase("OAuth2Block")
        'OAuth2 Block'
    """
    text = text[:_MAX_CAMELCASE_INPUT_LEN]
    # Step 1: insert space between lowercase/digit and uppercase: "camelCase" -> "camel Case"
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    # Step 2: insert space between uppercase run (2+ chars) and uppercase+lowercase.
    # Using {2,} instead of + prevents splitting single-char sequences like "OAuth2Block".
    text = re.sub(r"([A-Z]{2,})([A-Z][a-z])", r"\1 \2", text)
    return text
